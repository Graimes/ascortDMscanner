Данный модуль позволяет сканировать ДМ коды с помощью веб-камеры
Используются библиотека libdmtx в py обёртке
Для работы необходимо установить связаные модули
pip install pylibdmtx
pip install opencv-python
pip install pyperclip

Распространяется без поддержки и каких-либо гарантий. Доступно
коммерческое и не коммерческое использование и вообще любое 
использование при условии сохранения авторства. 

Примеры использования: 
    ascortDmScanner // открыть программу с параметрами по умолчанию и показывать на экране режим сканирования 
    ascortDmScanner --camID=1 --clipboard=True // Использовать камеру с номером 1 и скопировать в буфер обмена, как только будет найден УИН 
    ascortDmScanner --readFromFile='qr.jpg' --resultFile='result.txt'  // прочитать файл и сохранить УИН в файл result.txt''' 

Myagkov Anton @ wow1c
a@wow1c.com, 2023