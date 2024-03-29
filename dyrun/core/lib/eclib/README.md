<img src="https://github.com/KaoruTeranishi/EncryptedControl/blob/master/logo.png?raw=true" align="center" width="400" alt="header pic"/>

# ECLib

This is a Python library for numerical simulation of encrypted control.

# Encrypted control

Cybersecurity is a critical issue of networked control systems in a modern society.
Encrypted control is a novel concept of control using cryptographic tools for secure computation, such as homomorphic encryption and secret sharing.
ECLib helps researchers and students to implement their new idea of encrypted control using homomorphic encryption.

# Supported encryption schemes
- ElGamal<sup> [1]</sup>
- Dynamic-key ElGamal<sup> [2]</sup>
- Paillier<sup> [3]</sup>
- Regev (LWE)<sup> [4]</sup>
- GSW<sup> [5]</sup>
- GSW-LWE<sup> [6]</sup>

# Installation

Run pip command on your terminal.

`pip install eclib`

# Usage

See [tutorial_slide_1.pdf](https://github.com/KaoruTeranishi/EncryptedControl/blob/master/doc/tutorial_slide_1.pdf), [tutorial_slide_2.pdf](https://github.com/KaoruTeranishi/EncryptedControl/blob/master/doc/tutorial_slide_2.pdf), and [example codes](https://github.com/KaoruTeranishi/EncryptedControl/tree/master/examples).

# License

BSD 3-Clause License

Copyright (c) 2019, Kaoru Teranishi, Kimilab, The University of Electro-Communications
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author

Kaoru Teranishi
- E-mail: teranishi (at) uec.ac.jp
- Homepage: [https://kaoruteranishi.xyz](https://kaoruteranishi.xyz)

# References

[1] T. Elgamal, "A public key cryptosystem and a signature scheme based on discrete logarithms," IEEE Transactions on Information Theory, vol. 31, no. 4, pp. 469–472, 1985.

[2] K. Teranishi, T. Sadamoto, A. Chakrabortty, and K. Kogiso, "Designing optimal key lengths and control laws for encrypted control systems based on sample identifying complexity and deciphering time," IEEE Transactions on Automatic Control (Early access)

[3] P. Paillier, "Public-key cryptosystems based on composite degree residuosity classes," in International Conference on Theory and Application of Cryptographic Techniques, 1999, pp. 223–238.

[4] O. Regev, "On lattices, learning with errors, random linear codes, and cryptography," Journal of the ACM, vol. 56, no. 6, pp. 1-40, 2009.

[5] C. Gentry, A. Sahai, and B. Waters, "Homomorphic encryption from learning with errors: Conceptually-simpler, asymptotically-faster, attribute-based," Cryptology ePrint Archive, Paper 2013/340, 2013.

[6] J. Kim, H. Shim, and K. Han, "Dynamic controller that operates over homomorphically encrypted data for infinite time horizon," IEEE Transactions on Automatic Control, vol. 68, no. 2, pp. 660–672, 2023.