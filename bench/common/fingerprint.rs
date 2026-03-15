/// FNV1a-64 hash of a byte slice.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// FNV1a-64 hash of a UTF-8 string, returned as a zero-padded 16-char hex string.
pub fn fnv1a64_hex(data: &str) -> String {
    format!("{:016x}", fnv1a64(data.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_offset_basis() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325u64);
    }

    #[test]
    fn deterministic_across_calls() {
        let a = fnv1a64(b"hello");
        let b = fnv1a64(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn different_input_different_hash() {
        assert_ne!(fnv1a64(b"hello"), fnv1a64(b"world"));
    }

    #[test]
    fn hex_returns_16_char_string() {
        let h = fnv1a64_hex("hello");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hex_deterministic() {
        assert_eq!(fnv1a64_hex("hello"), fnv1a64_hex("hello"));
    }
}
