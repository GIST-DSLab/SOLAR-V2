"""과제 하나를 자식 프로세스에서 돌린다.

호출 단위로 RLIMIT_AS만 걸면, 폭주한 maker가 MemoryError로 잡히고 나서도
주소공간이 상한에 붙어 있어 그 다음 파일 읽기가 또 터진다 -- 실제로 그랬다.
프로세스를 끊으면 그 상태까지 같이 사라진다.

fork는 쓰기할 때만 복사하므로 400번 해도 비용이 거의 없다. 자식이 상한을
넘기거나 시간을 넘기면 죽고, 부모는 "이 과제는 못 쟀다"로 받아 계속 간다.
"""
import os, pickle, resource, signal, sys


def run(fn, *args, gib=3.0, seconds=300, **kwargs):
    """fn(*args)을 자식에서 돌려 결과를 돌려준다. 실패하면 (None, 사유)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            cap = int(gib * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            signal.alarm(seconds)
            out = fn(*args, **kwargs)
            with os.fdopen(w, "wb") as f:
                pickle.dump(out, f)
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(w)
    buf = b""
    with os.fdopen(r, "rb") as f:
        try:
            buf = f.read()
        except Exception:
            buf = b""
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        return None, f"signal {os.WTERMSIG(status)}"
    if os.WEXITSTATUS(status) != 0 or not buf:
        return None, "failed"
    try:
        return pickle.loads(buf), None
    except Exception:
        return None, "unpickle"
