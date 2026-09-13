runtime/
├── ir.h
├── runtime_attr.hpp        // RuntimeAttribute（权重）类声明
├── runtime_datatype.hpp    // 数据类型枚举
├── runtime_ir.hpp          // ✅【核心】RuntimeOperator、RuntimeOperand 结构体定义（UML图里的类）
├── runtime_op.hpp
├── runtime_operand.hpp     // RuntimeOperand单独声明
├── runtime_parameter.hpp   // RuntimeParameter基类+子类（超参）
└── store_zip.hpp
source/
├── ir.cpp
├── load_data.cpp
├── runtime_attr.cpp
├── runtime_ir.cpp          // ✅【重点实现】InitGraphOperatorsInput / InitGraphOperatorsOutput 就在这里！PNNX IR -> Runtime图的转换逻辑
├── runtime_op.cpp
├── store_zip.cpp
├── tensor_utils.cpp
└── tensor.cpp

runtime_operator
- 我叫什么名字
- 我是什么类型（Conv2d）
- 我的输入张量是谁、输出给谁
- 我的超参（stride/padding）、权重（weight/bias）在哪里