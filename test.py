import os
import numpy as np
import tensorflow as tf
from VisualTCAV import LocalVisualTCAV, GlobalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet_v2
import PIL.Image

# 注意：如果环境中 tensorflow.keras 引用有问题，可以尝试：
# from tensorflow.python.keras.applications.resnet_v2 import preprocess_input

# 为了演示核心过程，我们创建一个简化的测试脚本
# 它可以帮助你调试并理解 VisualTCAV 的计算流

def main():
    model_name = "ResNet50V2"
    graph_filename = "ResNet50V2-architecture-and-weights-compiled.h5"
    label_filename = "ResNet50V2-imagenet-classes.txt"
    
    # 检查模型文件是否存在，如果不存在则打印提示
    model_path = f"VisualTCAV/models/{model_name}/{graph_filename}"
    if not os.path.exists(model_path):
        print(f"警告: 找不到模型文件 {model_path}。")
        print("本脚本将演示如何初始化和调用核心方法。")
        # 这里我们可以退出或者尝试继续（如果只是为了看代码逻辑）
    
    # 2. 准备数据结构
    # test_image_filename 必须是 test_images 目录下的一张图片文件（而不是目录）
    # LocalVisualTCAV 内部会执行 PIL.Image.open(test_images_dir/test_image)
    test_image = "zebra.jpg"
    test_image_path = f"VisualTCAV/test_images/{test_image}"
    if not os.path.exists(test_image_path):
        print(f"警告: 找不到测试图片 {test_image_path}。")

    print("--- 步骤 1: 初始化 LocalVisualTCAV ---")
    try:
        local_visual_tcav = LocalVisualTCAV(
            test_image_filename=test_image,
            n_classes=3,
            m_steps=5, # 为了快速调试，减少步数
            batch_size=10,
            model=Model(
                model_name=model_name,
                graph_path_filename=graph_filename,
                label_path_filename=label_filename,
                preprocessing_function=preprocess_resnet_v2,
                max_examples=10 # 减少样本量方便调试
            )
        )
    except Exception as e:
        print(f"初始化失败 (可能是缺少模型或图像): {e}")
        return

    # 3. 设置分析的层和概念
    print("\n--- 步骤 2: 设置层和概念 ---")
    local_visual_tcav.setLayers(layer_names=["conv4_block6_out"])
    local_visual_tcav.setConcepts(concept_names=["striped"])

    # 4. 预测
    print("\n--- 步骤 3: 运行预测 ---")
    try:
        # predict() 会调用 model_wrapper.get_predictions
        preds = local_visual_tcav.predict()
        preds.info()
    except Exception as e:
        print(f"预测失败: {e}")

    # 5. 核心解释过程
    print("\n--- 步骤 4: 运行解释 (核心计算) ---")
    # explain() 内部包含以下关键步骤：
    # a. 计算随机激活 (_compute_random_activations)
    # b. 获取测试图片的特征图 (get_feature_maps)
    # c. 计算 CAV (_compute_cavs) -> 计算概念方向
    # d. 计算概念图 (concept_map) -> 激活与 CAV 方向的点积
    # e. 计算集成梯度 (_compute_integrated_gradients)
    # f. 计算归因 (attributions) -> 梯度、特征图与概念图的结合
    try:
        local_visual_tcav.explain(cache_cav=False, cache_random=False)
        print("解释计算完成。")
    except Exception as e:
        print(f"解释失败: {e}")
        print("\n调试提示: 核心计算过程在 VisualTCAV.py 的 LocalVisualTCAV.explain 方法中。")
        print("关键点如下：")
        print("1. CAV 计算: 行 262 (_compute_cavs)")
        print("2. 概念图生成: 行 395 (tf.multiply(concept_layer.cav.direction, feature_maps))")
        print("3. 集成梯度: 行 203 (_compute_integrated_gradients)")
        print("4. 最终归因掩码: 行 463 (tf.multiply(attributions, concept_map))")

    # 6. 可视化
    print("\n--- 步骤 5: 绘图 ---")
    try:
        import matplotlib.pyplot as plt
        local_visual_tcav.plot()
        # 把生成的所有图保存到文件，便于直接查看（替换了噪声概念图后不再是噪声）
        for i, num in enumerate(plt.get_fignums()):
            out_path = f"plot_output_{i}.png"
            plt.figure(num).savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"已保存绘图结果: {out_path}")
    except Exception as e:
        print(f"绘图失败: {e}")

    # 7. 全局分析与柱状图
    global_main(
        model_name=model_name,
        graph_filename=graph_filename,
        label_filename=label_filename,
    )

def global_main(model_name, graph_filename, label_filename):
    """使用 GlobalVisualTCAV 对一整个类别的图片进行分析，并绘制概念归因柱状图。"""

    print("\n=== 全局分析 (GlobalVisualTCAV) ===")

    # GlobalVisualTCAV 需要一个目标类别 (target_class) 以及一个存放该类别多张测试图片的文件夹
    # (test_images_folder)。该文件夹位于 test_images 目录下。
    target_class = "zebra"
    test_images_folder = "zebra"

    print("\n--- 步骤 G1: 初始化 GlobalVisualTCAV ---")
    try:
        global_visual_tcav = GlobalVisualTCAV(
            target_class=target_class,
            test_images_folder=test_images_folder,
            m_steps=5,  # 为了快速调试，减少步数
            batch_size=10,
            model=Model(
                model_name=model_name,
                graph_path_filename=graph_filename,
                label_path_filename=label_filename,
                preprocessing_function=preprocess_resnet_v2,
                max_examples=10,  # 减少样本量方便调试
            ),
        )
    except Exception as e:
        print(f"全局初始化失败 (可能是缺少模型或图像): {e}")
        return

    # 设置分析的层和概念
    print("\n--- 步骤 G2: 设置层和概念 ---")
    global_visual_tcav.setLayers(layer_names=["conv4_block6_out"])
    global_visual_tcav.setConcepts(concept_names=["striped"])

    # 核心解释过程：对整个类别的所有图片计算归因
    print("\n--- 步骤 G3: 运行全局解释 (核心计算) ---")
    try:
        global_visual_tcav.explain(cache_cav=False, cache_random=False)
        print("全局解释计算完成。")
    except Exception as e:
        print(f"全局解释失败: {e}")
        return

    # 打印统计信息
    print("\n--- 步骤 G4: 打印统计信息 ---")
    try:
        global_visual_tcav.statsInfo()
    except Exception as e:
        print(f"打印统计信息失败: {e}")

    # 绘制柱状图
    print("\n--- 步骤 G5: 绘制柱状图 ---")
    try:
        import matplotlib.pyplot as plt
        existing_fignums = set(plt.get_fignums())
        global_visual_tcav.plot()
        # 仅保存全局分析新生成的图（柱状图）
        new_fignums = [num for num in plt.get_fignums() if num not in existing_fignums]
        for i, num in enumerate(new_fignums):
            out_path = f"global_plot_output_{i}.png"
            plt.figure(num).savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"已保存全局柱状图结果: {out_path}")
    except Exception as e:
        print(f"全局绘图失败: {e}")

if __name__ == "__main__":
    main()
