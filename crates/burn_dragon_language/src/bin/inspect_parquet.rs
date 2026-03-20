use std::env;
use std::fs::File;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, ListAccessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .expect("usage: inspect_parquet <path-to-parquet>");
    let file = File::open(&path)?;
    let reader = SerializedFileReader::new(file)?;
    let mut rows = reader.get_row_iter(None)?;
    let row = rows
        .next()
        .transpose()?
        .expect("parquet file must contain at least one row");

    println!("path={path}");
    println!("row_len={}", row.len());
    for (idx, (name, field)) in row.get_column_iter().enumerate() {
        let kind = match field {
            Field::ListInternal(_) => "list",
            _ => "scalar",
        };
        println!("idx={idx} name={name} kind={kind}");
        match field {
            Field::ListInternal(list) => {
                let preview_len = list.len().min(8);
                let mut preview = Vec::with_capacity(preview_len);
                for i in 0..preview_len {
                    if let Ok(value) = list.get_long(i) {
                        preview.push(value.to_string());
                    } else if let Ok(value) = list.get_int(i) {
                        preview.push(value.to_string());
                    } else {
                        preview.push(format!("<{:?}>", field));
                        break;
                    }
                }
                println!("  list_len={} preview=[{}]", list.len(), preview.join(", "));
            }
            _ => println!("  value={field}"),
        }
    }

    Ok(())
}
