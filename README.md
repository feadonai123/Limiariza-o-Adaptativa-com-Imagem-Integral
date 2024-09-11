# Como rodar?

Executar no root do projeto o comando:
```
python index.py
```

Alterar `linha 160` para definir qual imagem será processada. Ex:
Para processar a imagem "teste.png":
```
imageFile = 'teste.png'
```
Para processar a imagem "teste2.png":
```
imageFile = 'teste2.png'
```

Após executar o script, será criados novos arquivos no root do projeto, com os seguintes formatos:
```
{NOME_ARQUIVO}_integral.jpg
{NOME_ARQUIVO}_otsu.jpg
{NOME_ARQUIVO}_adaptive.jpg
{NOME_ARQUIVO}_wellners.jpg
```
