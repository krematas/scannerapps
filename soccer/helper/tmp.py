import scannerpy
import cv2
from scannerpy import Database, DeviceType, Job, ColumnType, FrameType
import pickle


def test_python_source(db):
    # Write test files
    py_data = [{'{:d}'.format(i): i} for i in range(4)]

    data = db.sources.Python()
    pass_data = db.ops.Pass(input=data)
    output_op = db.sinks.Column(columns={'dict': pass_data})
    job = Job(op_args={
        data: {
            'data': pickle.dumps(py_data)
        },
        output_op: 'test_python_source',
    })

    tables = db.run(output_op, [job], force=True, show_progress=False)

    num_rows = 0
    for i, buf in enumerate(tables[0].column('dict').load()):
        d = pickle.loads(buf)
        assert d['{:d}'.format(i)] == i
        num_rows += 1
    assert num_rows == 4


db = Database()
test_python_source(db)
