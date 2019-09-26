#!/usr/bin/env python

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PREFIX = "../data/"


def main():
    convert("train_transaction.csv.zip",
            "train_transaction.gzip.pq")

    convert("train_identity.csv.zip",
            "train_identity.gzip.pq")

    convert("test_transaction.csv.zip",
            "test_transaction.gzip.pq")

    convert("test_identity.csv.zip",
            "test_identity.gzip.pq")

    return 1


def convert(input, output):
    input, output = PREFIX + input, PREFIX + output
    pq.write_table(pa.Table.from_pandas(pd.read_csv(input)),
                   output,
                   compression="gzip")
    print(output, "is created.")


if __name__ == "__main__":
    main()
