import scannerpy
import os
import scanner.metadata_pb2 as metadata_types
import scannerpy._python as bindings


mp = bindings.default_machine_params()
mp_proto = metadata_types.MachineParameters()
mp_proto.ParseFromString(mp)
mp_proto.num_load_workers = 32
mp = mp_proto.SerializeToString()

scannerpy.start_worker(
    '{}:{}'.format(os.environ['SCANNER_MASTER_SERVICE_HOST'],
                   os.environ['SCANNER_MASTER_SERVICE_PORT']),
    block=True,
    watchdog=False,
    port=5002,
    machine_params=mp)
