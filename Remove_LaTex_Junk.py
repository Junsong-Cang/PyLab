import os

path = os.getcwd()
files = os.listdir(path)
junk_files = []
pdf_files = []

for f in files:
    suffix = ''
    for idx in [-4, -3, -2, -1]:
        suffix += f[idx]
    if suffix in ['tore', '.out', '.log', '.aux', 'x.gz', '.bbl', '.blg']:
        junk_files.append(f)
    if f == '.DS_Store':
        junk_files.append(f)
    if suffix == '.pdf':
        pdf_files.append(f)

print("当前目录：", path)
if len(junk_files) == 0:
    print('无垃圾文件')
else:
    print('垃圾文件：')
    for f in junk_files:
        print("    "+f)

if len(pdf_files) == 0:
    print('无pdf文件')
else:
    print('pdf文件：')
    for f in pdf_files:
        print("    "+f)
if (len(junk_files) == 0) and (len(pdf_files) == 0):
    print('目录无垃圾和pdf文件，无需后续操作')
else:
    action = int(eval(input('请输入操作：0-取消，1-删除垃圾，2-删除包括pdf在内的所有垃圾\n')))
    if not action in [0, 1, 2]:
        raise Exception("错误：未知输入")
    if action in [1, 2]:
        for f in junk_files:
            os.remove(path+'/'+f)
    if action == 2:
        for f in pdf_files:
            os.remove(path+'/'+f)
