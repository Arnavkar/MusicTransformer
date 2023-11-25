from multiprocessing import Process, Queue, shared_memory, managers

class ShmArray(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, shm=None):
        obj = super(ShmArray, cls).__new__(cls, shape, dtype,
                                           buffer, offset, strides,
                                           order)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)



def shared_mem_multiprocessing(sequence, workers=2, queue_max_size=16):

    class ErasingSharedMemory(shared_memory.SharedMemory):

        def __del__(self):
            super(ErasingSharedMemory, self).__del__()
            self.unlink()

    queue = Queue(maxsize=queue_max_size)
    manager = managers.SharedMemoryManager()
    manager.start()

    def worker(sequence, idxs):
        for i in idxs:
            x, y = sequence[i]

            shm = manager.SharedMemory(size=x.nbytes + y.nbytes)
            a = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf, offset=0)
            b = np.ndarray(y.shape, dtype=y.dtype, buffer=shm.buf, offset=x.nbytes)

            a[:] = x[:]
            b[:] = y[:]
            queue.put((a.shape, a.dtype, b.shape, b.dtype, shm.name))
            shm.close()
            del shm

    idxs = np.array_split(np.arange(len(sequence)), workers)
    args = zip([sequence] * workers, idxs)
    processes = [Process(target=worker, args=(s, i)) for s, i in args]
    _ = [p.start() for p in processes]

    try:
        for i in range(len(sequence)):
            x_shape, x_dtype, y_shape, y_dtype, shm_name = queue.get(block=True)
            existing_shm = ErasingSharedMemory(name=shm_name)
            x = ShmArray(x_shape, dtype=x_dtype, buffer=existing_shm.buf, offset=0, shm=existing_shm)
            y = ShmArray(y_shape, dtype=y_dtype, buffer=existing_shm.buf, offset=x.nbytes, shm=existing_shm)
            yield x, y
            # Memory will be automatically deleted when gc is triggered
    finally:
        print("Closing all the processed")
        _ = [p.terminate() for p in processes]
        print("Joining all the processed")
        _ = [p.join() for p in processes]
        queue.close()
        manager.shutdown()
        manager.join()