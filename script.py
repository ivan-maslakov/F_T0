from cirq import protocols
from cirq.testing import gate_features

def extracter(block_file_name, num_ext_qubits):
    '''приводит данные файлов прекодинга, посткодинга и операции к формату входных данных функции block_constructer (то есть к массиву последовательных гейтов)
    Входные данные: block_file_name - имя файла,
    num_ext_qubits - кол-во кубитов во ВСЕЙ цепочке
    Выходные данные: массив наименований гейтов, кубитов, на которые они действуют и параметров
    '''

def block_constructer(cirquit, total_num_qub, gates_array):
    '''дописывает в данную цепочку (в ее конец) !кутритные! cirq-гейты данного блока (то есть создает участок цепочки на 3 + num_ext_qub кутритов)
    Входные данные: cirquit - цепочка, в которую дописываем,
    total_num_qubs - ОБЩЕЕ кол-во кубитов во всей цепочке
    gates_array - массив кортежей из трех элементов: (gate_name, qub, params),
    например [('MS', (a, b), (i, j, k, l, theta, phi)), ('R', (c), (m,n, phi)), ...]
    Выходные данные: дописанная цепочка
    КОММЕНТАРИЙ: зная какой именно гейт, мы знаем, какие и СКОЛЬКО в нем параметров и на СКОЛЬКО кубитов он действует
    '''

def encoding_constructer(total_num_qubs, cirquit):
    '''дописывает в данную цепочку (в ее конец) !кутритные! cirq-гейты кодировки
    Входные данные: total_num_qubs - ОБЩЕЕ кол-во кубитов во всей цепочке,
    cirquit - цепочка, в которую дописываем
    Выходные данные: дописанная цепочка
    '''

def dencoding_constructer(total_num_qubs, cirquit):
    '''дописывает в данную цепочку (в ее конец) !кутритные! cirq-гейты декодировки
    Входные данные: total_num_qubs - ОБЩЕЕ кол-во кубитов во всей цепочке,
    cirquit - цепочка, в которую дописываем
    Выходные данные: дописанная цепочка
    '''

def messurements(total_num_qubs, cirquit):
    '''дописывает в данную цепочку (в ее конец) !кутритные! cirq-гейты декодировки
        Входные данные: total_num_qubs - ОБЩЕЕ кол-во кубитов во всей цепочке,
        cirquit - цепочка, в которую дописываем
        Выходные данные: дописанная цепочка
        '''

def full_cirq_constructor(total_num_qubs, precoding_file_name, operation_file_name, postcoding_file_name):
    '''записывет цепочку cirq, состоящую из прекодинга, кодировки, операции, декодировки, пост кодинга.
    Последовательно применяет сначала extracter_Ы, потом block_constructer_Ы ну и encoding_constructer, dencoding_constructer
    Входные данные: total_num_qubs - ОБЩЕЕ кол-во кубитов во всей цепочке,
    precoding_file_name, operation_file_name, postcoding_file_name - имена файлов с блоками операций
    Выходные данные: готовая цепочка
    '''

# определение !классов! КУТРИТНЫХ MS, R гейтов
# добавить функции CX_ijk, X_ij итд (о всеми параметрами)