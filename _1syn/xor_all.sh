for len in 16 #32 64 128 256 512 1024 2048
do
#  python3 xor_generation.py --length $len
  python3 xor_main.py --length $len --model CDIL
  python3 xor_main.py --length $len --model CNN
  python3 xor_main.py --length $len --model TCN
  python3 xor_main.py --length $len --model Deformable
  python3 xor_main.py --length $len --model GRU
  python3 xor_main.py --length $len --model LSTM
  python3 xor_main.py --length $len --model Transformer
  python3 xor_main.py --length $len --model Linformer
  python3 xor_main.py --length $len --model Performer
done
