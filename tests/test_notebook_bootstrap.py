"""Validate the self-contained notebook bundle without training or installing ML."""
import ast
import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import nbformat

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(PROJECT / 'model_comparison.ipynb', as_version=4)


def bootstrap_namespace():
    cell = next(cell for cell in NOTEBOOK.cells
                if cell.cell_type == 'code' and 'def extract_embedded_project' in cell.source)
    tree = ast.parse(cell.source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'extract_embedded_project', 'complete_project'}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(name, ast.Name)
                and name.id.startswith('EMBEDDED_') for name in node.targets):
            selected.append(node)
    namespace = {'Path': Path, 'base64': base64, 'hashlib': hashlib, 'io': io, 'zipfile': zipfile}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook-bootstrap>', 'exec'), namespace)
    return namespace


class NotebookBootstrapTests(unittest.TestCase):
    def test_bundle_contains_exact_canonical_sources_and_reusable_extraction(self):
        namespace = bootstrap_namespace()
        with tempfile.TemporaryDirectory() as temporary:
            root = namespace['extract_embedded_project'](Path(temporary) / 'project')
            self.assertTrue(namespace['complete_project'](root))
            for name, expected in namespace['EMBEDDED_FILE_SHA256'].items():
                self.assertEqual((root / name).read_bytes(), (PROJECT / name).read_bytes(), name)
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
            self.assertEqual(namespace['extract_embedded_project'](root), root)

    def test_existing_changed_project_is_not_overwritten(self):
        namespace = bootstrap_namespace()
        with tempfile.TemporaryDirectory() as temporary:
            root = namespace['extract_embedded_project'](Path(temporary) / 'project')
            changed = root / 'requirements.txt'
            changed.write_text('user modification\n', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                namespace['extract_embedded_project'](root)
            self.assertEqual(changed.read_text(), 'user modification\n')

    def test_tampered_archive_is_rejected_before_extraction(self):
        namespace = bootstrap_namespace()
        namespace['EMBEDDED_PROJECT_SHA256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'project'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                namespace['extract_embedded_project'](destination)
            self.assertFalse(destination.exists())

    def test_path_escape_is_rejected_even_with_matching_checksums(self):
        namespace = bootstrap_namespace()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('../outside.txt', b'not allowed')
        raw = buffer.getvalue()
        namespace.update(EMBEDDED_PROJECT_B64=base64.b64encode(raw).decode(),
                         EMBEDDED_PROJECT_SHA256=hashlib.sha256(raw).hexdigest(),
                         EMBEDDED_FILE_SHA256={'../outside.txt': hashlib.sha256(b'not allowed').hexdigest()})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                namespace['extract_embedded_project'](Path(temporary) / 'project')
            self.assertFalse((Path(temporary) / 'outside.txt').exists())

    def test_nested_runtime_probe_has_valid_python(self):
        cell = next(cell for cell in NOTEBOOK.cells
                    if cell.cell_type == 'code' and 'runtime_probe = ' in cell.source)
        tree = ast.parse(cell.source)
        probe = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'runtime_probe'
                             for target in node.targets))
        ast.parse(probe)

    def test_portable_notebook_has_no_execution_claims(self):
        nbformat.validate(NOTEBOOK)
        for cell in NOTEBOOK.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])


if __name__ == '__main__':
    unittest.main()
