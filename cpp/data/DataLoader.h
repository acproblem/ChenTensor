//
// Created by Lenovo on 2023/9/1.
// 数据集加载器类。
//

#ifndef CHENTENSOR_DATALOADER_H
#define CHENTENSOR_DATALOADER_H

#include "Dataset.h"
#include "functional/__ConcatenateFunc__.h"
#include <vector>
#include <algorithm>


/// 数据集加载器类。
template<typename TX, typename TY>
class DataLoader {
private:
    std::vector<std::tuple<Tensor<TX>, Tensor<TY>>> data;
    int batch_size, pos;
public:
    DataLoader(Dataset<TX, TY> &dataset, int batch_size, bool shuffle = true)
            : batch_size(batch_size), pos(0) {
        if (batch_size <= 0)
            throw std::runtime_error("The parameter `batch_size` must be greater than zero,");

        // 初始化一个加载数据的次序
        std::vector<unsigned int> idxs(dataset.size());
        for (int i = 0; i < idxs.size(); i++)
            idxs[i] = i;
        if (shuffle)  // 打乱顺序
            std::random_shuffle(idxs.begin(), idxs.end());

        // 获取数据
        data = std::vector<std::tuple<Tensor<TX>, Tensor<TY>>>(
                idxs.size() / batch_size + (idxs.size() % batch_size != 0));
        int j = 0;
        for (int i = 0; i < data.size(); i++) {
            int lastPos = std::min((i + 1) * batch_size, static_cast<int>(idxs.size()));
            std::vector<Tensor<TX>> X;
            std::vector<Tensor<TY>> Y;
            for (; j < lastPos; j++) {
                auto tdata = dataset[j];
                X.push_back(std::get<0>(tdata));
                Y.push_back(std::get<1>(tdata));
            }
            this->data[i] = std::make_tuple(union_tensor(X), union_tensor(Y));
        }
    }

    bool hasNext() const {
        return pos < data.size();
    }

    const std::tuple<Tensor<TX>, Tensor<TY>> &next() {
        return data[pos++];
    }

    int size() const {
        return data.size();
    }

    /// 仅仅让迭代重新开始
    void reset() {
        pos = 0;
    }
};

#endif //CHENTENSOR_DATALOADER_H
