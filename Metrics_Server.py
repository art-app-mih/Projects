# реализация сервера для тестирования метода get по заданию - Клиент для отправки метрик
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# asyncio, tcp сервер
import asyncio
from asyncio import transports


class ServerProtocol(asyncio.Protocol):
    login: str = None
    server: 'Server'
    transport: transports.Transport

    def __init__(self, server: 'Server'):
        self.server = server

    def data_received(self, data: bytes):
        print(data)

        decoded = data.decode()
        list_of_data = []

        if decoded.startswith('put'):
            info_metrics = decoded.replace("put", "").replace("\r\n", "")
            list_of_data = info_metrics.strip().split()
            self.server.add_metrics(list_of_data[0], int(list_of_data[1]))
            print("Загружена новая метрика:")
            print(list_of_data[0], ":", int(list_of_data[1]))
        elif decoded.startswith('get'):
            info_metrics = decoded.replace("get", "").replace("\r\n", "")
            list_of_data = info_metrics.strip().split()
            if list_of_data[0] == '*':
                for now in self.server.metrics:
                    self.transport.write(self.server.metrics[now])
            else:
                data_to_transport = self.server.metrics[list_of_data[0]]
                self.transport.write(data_to_transport)
        else:
            self.transport.write('Введена недоступная метрика')

    def connection_made(self, transport: transports.Transport):
        self.server.clients.append(self)
        self.transport = transport
        print("Пришел новый клиент")

    def connection_lost(self, exception):
        self.server.clients.remove(self)
        print("Клиент вышел")

    def send_message(self, content: str):
        message = f"{content}\n"


class Server:
    clients: list
    history_messages = []
    metrics = {}

    def __init__(self):
        self.clients = []

    def build_protocol(self):
        return ServerProtocol(self)

    def add_metrics(self, name_of_metrics:str, value:int):
        self.metrics[f'{name_of_metrics}'] = value
        return self.metrics

    async def start(self):
        loop = asyncio.get_running_loop()

        coroutine = await loop.create_server(
            self.build_protocol,
            '127.0.0.1',
            8888
        )

        print("Сервер запущен ...")

        await coroutine.serve_forever()


process = Server()
try:
    asyncio.run(process.start())
except KeyboardInterrupt:
    print("Сервер остановлен вручную")