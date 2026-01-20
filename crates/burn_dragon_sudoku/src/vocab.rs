use anyhow::{Result, anyhow};

pub const GRID_LEN: usize = 81;
pub const VOCAB_SIZE: usize = 10;

#[derive(Clone, Debug, Default)]
pub struct SudokuVocab;

impl SudokuVocab {
    pub fn encode_grid(text: &str) -> Result<Vec<u8>> {
        let mut values = Vec::with_capacity(GRID_LEN);
        for ch in text.chars() {
            if ch.is_ascii_digit() {
                values.push(ch.to_digit(10).unwrap() as u8);
            } else if ch == '.' || ch == '_' {
                values.push(0);
            }
        }

        if values.len() != GRID_LEN {
            return Err(anyhow!(
                "expected {} cells but parsed {}",
                GRID_LEN,
                values.len()
            ));
        }

        Ok(values)
    }

    pub fn decode_grid(cells: &[u8]) -> Result<String> {
        if cells.len() != GRID_LEN {
            return Err(anyhow!(
                "expected {} cells but got {}",
                GRID_LEN,
                cells.len()
            ));
        }
        let mut out = String::with_capacity(GRID_LEN);
        for &value in cells {
            if value > 9 {
                return Err(anyhow!("invalid cell value {}", value));
            }
            out.push(char::from(b'0' + value));
        }
        Ok(out)
    }

    pub fn format_grid(cells: &[u8]) -> Result<String> {
        if cells.len() != GRID_LEN {
            return Err(anyhow!(
                "expected {} cells but got {}",
                GRID_LEN,
                cells.len()
            ));
        }
        let mut out = String::with_capacity(GRID_LEN + 16);
        for row in 0..9 {
            let start = row * 9;
            let end = start + 9;
            for &value in &cells[start..end] {
                let ch = char::from(b'0' + value.min(9));
                out.push(ch);
                out.push(' ');
            }
            out.pop();
            if row + 1 != 9 {
                out.push('\n');
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        let grid = "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
        let tokens = SudokuVocab::encode_grid(grid).expect("encode");
        let decoded = SudokuVocab::decode_grid(&tokens).expect("decode");
        assert_eq!(decoded, grid);
    }

    #[test]
    fn format_grid_is_9x9() {
        let grid = "0".repeat(GRID_LEN);
        let tokens = SudokuVocab::encode_grid(&grid).expect("encode");
        let formatted = SudokuVocab::format_grid(&tokens).expect("format");
        let lines: Vec<&str> = formatted.split('\n').collect();
        assert_eq!(lines.len(), 9);
        assert!(lines.iter().all(|line| line.split(' ').count() == 9));
    }
}
