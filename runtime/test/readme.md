1）generate demo_net.pt
python create_demonet.py
2）generate .param and .bin
./pnxx demo_net.pt inputshape=[1,3,32,32]
3）complile pnnx_view.cpp
g++ pnnx_view.cpp ir.cpp store_zip.cpp -o pnnx_view
4）result
./pnnx_view