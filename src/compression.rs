//! Provides compression utilities for encoding records.
//!
//! This module has implementations of gzip, Snappy, as well as a noop compression format that
//! allows encoding and decoding records into a [`Record`](crate::records::Record).

use crate::protocol::buf::{ByteBuf, ByteBufMut};
use anyhow::Result;

#[cfg(feature = "gzip")]
mod gzip;
#[cfg(feature = "lz4")]
mod lz4;
mod none;
#[cfg(feature = "snappy")]
mod snappy;
#[cfg(feature = "zstd")]
mod zstd;

#[cfg(feature = "gzip")]
pub use gzip::Gzip;
#[cfg(feature = "lz4")]
pub use lz4::Lz4;
pub use none::None;
#[cfg(feature = "snappy")]
pub use snappy::Snappy;
#[cfg(feature = "zstd")]
pub use zstd::Zstd;

/// A trait for record compression algorithms.
pub trait Compressor<B: ByteBufMut> {
    /// Target buffer type for compression.
    type BufMut: ByteBufMut;
    /// Compresses into provided [`ByteBufMut`], with records encoded by `F` into `R`.
    fn compress<R, F>(buf: &mut B, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self::BufMut) -> Result<R>;
}

/// A trait for record decompression algorithms.
pub trait Decompressor<B: ByteBuf> {
    /// Target buffer type for decompression.
    type Buf: ByteBuf;
    /// Decompress records from `B` mapped using `F` into `R`.
    fn decompress<R, F>(buf: &mut B, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self::Buf) -> Result<R>;
}

/// Hard cap on the number of bytes a single record-batch decompression may produce.
///
/// Compression formats can inflate a tiny input into many gigabytes (a "decompression bomb"); the
/// decoder would otherwise allocate the full output before any size is validated, letting one
/// crafted batch exhaust memory. 256 MiB comfortably exceeds any legitimate Kafka batch (a broker's
/// `message.max.bytes` is typically ~1 MiB) while bounding the worst case. The per-format
/// `Decompressor` implementations stop and return an error once a single batch exceeds this.
pub const MAX_DECOMPRESSED_SIZE: usize = 256 * 1024 * 1024;

/// A [`std::io::Write`] adapter that fails once more than `limit` bytes have been written, so a
/// streaming decompressor cannot inflate without bound (see [`MAX_DECOMPRESSED_SIZE`]).
pub(crate) struct BoundedWriter<W> {
    inner: W,
    written: usize,
    limit: usize,
}

impl<W> BoundedWriter<W> {
    pub(crate) fn new(inner: W, limit: usize) -> Self {
        Self {
            inner,
            written: 0,
            limit,
        }
    }

    pub(crate) fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: std::io::Write> std::io::Write for BoundedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written = self.written.saturating_add(buf.len());
        if self.written > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "decompressed record batch exceeds the maximum allowed size",
            ));
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
