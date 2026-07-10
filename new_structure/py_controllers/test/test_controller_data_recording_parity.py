"""Exact parity tests for the extracted controller CSV helpers."""

import hashlib
import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from py_controllers.controller_data_recording import (  # noqa: E402
    build_full_csv_header,
    controller_csv_path,
    final_csv_column_names,
    final_csv_extra_header,
    project_row,
    requested_column_indices,
)


def schema_digest(columns):
    return hashlib.sha256("\n".join(columns).encode()).hexdigest()


class ControllerDataRecordingParityTest(unittest.TestCase):
    def test_full_header_exact_refactor_baseline(self):
        header = build_full_csv_header(7)
        self.assertEqual(len(header), 472)
        self.assertEqual(len(set(header)), 472)
        self.assertEqual(
            schema_digest(header),
            "5d5825a0e115b04db7d61388d41b65f6f1891705fb83a08212c65c6dbec7eeb7",
        )

    def test_final_schema_and_extra_header_exact_refactor_baseline(self):
        final_columns = final_csv_column_names()
        extra = final_csv_extra_header()
        self.assertEqual(len(final_columns), 159)
        self.assertEqual(
            schema_digest(final_columns),
            "266c28db48b1f6d547cbc329f1859bc74f87ee28f62dde2292207c8b435620bf",
        )
        self.assertEqual(len(extra), 7)
        self.assertEqual(
            schema_digest(extra),
            "b1b2e4bd4ba37861cf1fd61e8b1b604d2caef37eb64d980cf49a1431921b0c9d",
        )

    def test_projection_preserves_every_value_and_position(self):
        header = build_full_csv_header(7) + final_csv_extra_header()
        indices, missing = requested_column_indices(
            header, final_csv_column_names()
        )
        self.assertEqual(missing, [])
        values = [float(index) for index in range(len(header))]
        values[header.index("run_name")] = "synthetic"
        values[header.index("hist_db_nearest_distance")] = float("nan")
        values[header.index("hist_db_preflight_phase")] = None
        projected = project_row(values, indices)
        self.assertEqual(len(projected), len(final_csv_column_names()))
        for output_index, source_index in enumerate(indices):
            expected = values[source_index]
            actual = projected[output_index]
            if isinstance(expected, float) and math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)

    def test_missing_columns_and_input_row_are_not_mutated(self):
        header = ["a", "b"]
        row = [1.0, 2.0]
        indices, missing = requested_column_indices(header, ["b", "c", "a"])
        self.assertEqual(indices, [1, 0])
        self.assertEqual(missing, ["c"])
        self.assertEqual(project_row(row, indices), [2.0, 1.0])
        self.assertEqual(header, ["a", "b"])
        self.assertEqual(row, [1.0, 2.0])

    def test_output_path_semantics(self):
        output_dir, filename = controller_csv_path(
            "~/goal12-output", "nested/run-A"
        )
        self.assertEqual(output_dir, Path("~/goal12-output").expanduser())
        self.assertEqual(
            filename.name, "run-A_cartesian_impedance_controller_data.csv"
        )
        _, default_filename = controller_csv_path("/tmp/out", "")
        self.assertEqual(
            default_filename.name, "cartesian_impedance_controller_data.csv"
        )


if __name__ == "__main__":
    unittest.main()
