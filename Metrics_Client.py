# реализация Клиент для отправки метрик
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import time


class ClientError(BaseException):
    def __init__(self, text):
        self.txt = text


class Client:
    def __init__(self, ipv4_address, num_of_port, timeout=None):
        self.host = ipv4_address
        self.port = num_of_port
        self.timeout = timeout

    def get(self, name_of_metric):
        if 'cpu' in name_of_metric or 'memory' in name_of_metric or '*' in name_of_metric:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                try:
                    sock.sendall(name_of_metric.encode('utf8'))
                    data = sock.recv(1024)
                    print(data.decode('utf8'))
                except ClientError('Error of connection'):
                    return 0
        else:
            print('Nonexistent metric')

    def put(self, name_of_metric_and_cpu, value_of_metric,  timestamp=int(time.time())):
        with socket.create_connection((self.host, self.port), self.timeout) as sock:
            sock.settimeout(self.timeout)
            try:
                value_of_metric = int(value_of_metric)
                sock.sendall(name_of_metric_and_cpu.encode('utf8'), value_of_metric)
            except ClientError('Error of connection'):
                return 0
