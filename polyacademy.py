import multiprocessing as mp

import sys
from multiprocessing.connection import Listener, Client

address = ('localhost', 6000)


def client():
    while True:
        conn = Client(address, authkey=b'secret password')
        conn.send(1)
        s = conn.recv()
        if s==6:
            break
        print(s)
        conn.close()


def server():
    listener = Listener(address, authkey=b'secret password')

    queue = []

    while True:
        conn = listener.accept()
        queue.append((conn, conn.recv()))

        sss = 0
        if len(queue) >= 2:
            for conn, b in queue:
                sss += b

            for conn, b in queue:
                conn.send(sss)

            for conn, b in queue:
                conn.close()

            queue = []

    listener.close()

import time

if __name__ == '__main__':
    a = mp.Process(target=server)
    a.start()
    time.sleep(5)
    b = []
    for i in range(4):
        bi = mp.Process(target=client)
        bi.start()
        b.append(bi)