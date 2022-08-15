from linkefl.common.factory import messenger_factory
from linkefl.common.const import Const

if __name__ == '__main__':
    active_ip = 'localhost'
    active_port = 11111
    passive_ip = 'localhost'
    passive_port = 22222

    active_messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                         role=Const.ACTIVE_NAME,
                                         active_ip=active_ip,
                                         active_port=active_port,
                                         passive_ip=passive_ip,
                                         passive_port=passive_port)
    passive_messenger = messenger_factory(messenger_type=Const.FAST_SOCKET,
                                         role=Const.PASSIVE_NAME,
                                         active_ip=active_ip,
                                         active_port=active_port,
                                         passive_ip=passive_ip,
                                         passive_port=passive_port)

    data0 = [1, 2, 3, 4]
    data1 = {'a': 1, 'b': 2, 'c': 3}
    data2 = 'hello world'

    active_messenger.send(data0)
    active_messenger.send(data1)
    active_messenger.send(data2)

    recv_data0 = passive_messenger.recv()
    recv_data1 = passive_messenger.recv()
    recv_data2 = passive_messenger.recv()

    print(recv_data0, recv_data1, recv_data2)