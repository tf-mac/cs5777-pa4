import torch
import cs4787pa4
import time

torch.backends.cuda.matmul.allow_tf32 = False

if __name__ == '__main__':
    M = 3 * 1024
    N = 4 * 1024
    K = 5 * 1024

    print('float32')

    A = torch.randn(M,K,device='cuda')
    B = torch.randn(N,K,device='cuda')
    C = torch.zeros(M,N,device='cuda')
    C1 = torch.zeros(M,N,device='cuda')
    C2 = torch.zeros(M,N,device='cuda')
    C3 = torch.zeros(M,N,device='cuda')
    torch.cuda.synchronize()

    start = time.time()
    torch.mm(A,B.t(),out=C)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for torch.mm: {stop - start} s')

    start = time.time()
    cs4787pa4.sgemm_p1(A,B,C1)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for sgemm_p1: {stop - start} s')

    start = time.time()
    cs4787pa4.sgemm_p2(A,B,C2)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for sgemm_p2: {stop - start} s')

    ### uncomment this to test the tf32 tensorcore code
    # start = time.time()
    # cs4787pa4.sgemm_p3(A,B,C3)
    # torch.cuda.synchronize()
    # stop = time.time()
    # print(f'elapsed for sgemm_p3: {stop - start} s')

    Cd = A.double() @ B.double().t() # get high-precision result

    print(f'relative square error torch: {(C.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p1: {(C1.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p2: {(C2.double() - Cd).square().sum() / Cd.square().sum()}')
    # print(f'relative square error    p3: {(C3.double() - Cd).square().sum() / Cd.square().sum()}')


    print('float16')

    A = torch.randn(M,K,dtype=torch.float16,device='cuda')
    B = torch.randn(N,K,dtype=torch.float16,device='cuda')
    C = torch.zeros(M,N,dtype=torch.float16,device='cuda')
    C1 = torch.zeros(M,N,dtype=torch.float16,device='cuda')
    C2 = torch.zeros(M,N,dtype=torch.float16,device='cuda')
    C3 = torch.zeros(M,N,dtype=torch.float16,device='cuda')
    C4 = torch.zeros(M,N,dtype=torch.float32,device='cuda')
    torch.cuda.synchronize()

    start = time.time()
    torch.mm(A,B.t(),out=C)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for torch.mm: {stop - start} s')

    start = time.time()
    cs4787pa4.hgemm_p1(A,B,C1)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for hgemm_p1: {stop - start} s')

    start = time.time()
    cs4787pa4.hgemm_p2(A,B,C2)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for hgemm_p2: {stop - start} s')

    start = time.time()
    cs4787pa4.hgemm_p3(A,B,C3)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for hgemm_p3: {stop - start} s')

    start = time.time()
    cs4787pa4.hgemm_p4(A,B,C4)
    torch.cuda.synchronize()
    stop = time.time()
    print(f'elapsed for hgemm_p4: {stop - start} s')

    Cd = A.double() @ B.double().t() # get high-precision result

    print(f'relative square error torch: {(C.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p1: {(C1.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p2: {(C2.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p3: {(C3.double() - Cd).square().sum() / Cd.square().sum()}')
    print(f'relative square error    p4: {(C4.double() - Cd).square().sum() / Cd.square().sum()}')
