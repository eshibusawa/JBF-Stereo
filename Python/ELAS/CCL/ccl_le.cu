// This file is part of JBF-Stereo.
// Copyright (c) 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This file is a modified version of CCL <https://github.com/foota/ccl>, see MIT license below.

// The MIT License (MIT)
// Copyright (c) 2012 Noriyuki Futatsugi
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// Marathon Match - CCL - Label Equivalence

extern "C" __global__ void init_CCL(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[id] = id;
}

extern "C" __global__ void scanning(short D[], int L[], int R[], int* m, int N, int W, short th)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	short Did = D[id];
	int label = N;
	if (id - W >= 0 && abs(Did - D[id-W]) <= th) label = min(label, L[id-W]);
	if (id + W < N  && abs(Did - D[id+W]) <= th) label = min(label, L[id+W]);
	int r = id % W;
	if (r           && abs(Did - D[id-1]) <= th) label = min(label, L[id-1]);
	if (r + 1 != W  && abs(Did - D[id+1]) <= th) label = min(label, L[id+1]);

	if (label < L[id]) {
		R[L[id]] = label;
		*m = 0xffffffff;
	}
}

extern "C" __global__ void analysis(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	int label = L[id];
	int ref;
	if (label == id) {
		do { label = R[ref = label]; } while (ref ^ label);
		R[id] = label;
	}
}

extern "C" __global__ void labeling(int L[], int R[], int N)
{
	int id = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x + threadIdx.x;
	if (id >= N) return;

	L[id] = R[R[L[id]]];
}
