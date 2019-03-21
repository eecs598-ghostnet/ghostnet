import subprocess, re

def e2p(s):
    args = ['-q']
    args.append('-x')
    args.append('--sep=-')
    args.append('-m')
    args.append(s)
    return espeak_exe(args,sync=True)

def espeak_exe(args, sync=False):
    cmd = ['espeak-ng',
               '-b', '1', # UTF8 text encoding
               ]

    cmd.extend(args)
    p = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)

    res = iter(p.stdout.readline, b'')
    if not sync:
        p.stdout.close()
        if p.stderr:
            p.stderr.close()
        if p.stdin:
            p.stdin.close()
        return res

    res2 = ''
    for line in res:
        res2 += (line).decode("utf-8")

    p.stdout.close()
    if p.stderr:
        p.stderr.close()
    if p.stdin:
        p.stdin.close()
    p.wait()

    return res2

if __name__ == "__main__":
    output = e2p('All up in my phone lookin at pictures from||the other night <break>')
    print(output)
    output = re.sub("[',%=_|]",'',output)
    output = re.sub("_:", '', output)
    print(output)
