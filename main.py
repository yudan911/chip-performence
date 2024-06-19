import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import time
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from itertools import cycle
# 设置全局字体大小和线条宽度
rcParams['font.size'] = 20 # 默认字体大小为10，这里设置为19
rcParams['axes.linewidth'] = 1.5  # 默认线条宽度为0.8，这里加倍到1.5
rcParams['lines.linewidth'] = 2  # 默认折线图线条宽度，这里设置为2以加粗
fm.fontManager.addfont('times.ttf')
# 设置全局字体
plt.rcParams["font.family"] = ["Times New Roman"]
np.random.seed(42)

# 加载数据
file_path = 'ML数据更新.xlsx'
df = pd.read_excel(file_path)

feature_names = ['CD66b/CD63+EVs', 'miR-223-3p', 'miR-425-5p', 'CEA', 'CA199']
for col in feature_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

X = df[feature_names].values
y = df['Diagnosis'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 确保 "HC" 映射到 0, "GC" 映射到 1
classes = le.classes_
if 'HC' in classes and 'GC' in classes:
    index_HC, index_GC = classes.tolist().index('HC'), classes.tolist().index('GC')
    # 一步到位进行映射，确保"HC"为0，"GC"为1
    y_final = np.where(y_encoded == index_HC, 0, np.where(y_encoded == index_GC, 1, y_encoded))
    # 验证映射结果
    assert (y_final == 0).sum() == (y == 'HC').sum() and (y_final == 1).sum() == (y == 'GC').sum(), \
           "映射错误，'HC' 不全为 0 或 'GC' 不全为 1."
else:
    raise ValueError("Ensure that 'HC' and 'GC' are present in the labels.")


print(f"\nHC 映射后: {y_final[y == 'HC'][0]}, GC 映射后: {y_final[y == 'GC'][0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.3, random_state=42)
def safe_filename(feature):
    return feature.replace('/', '_').replace(' ', '_')
def evaluate_and_plot_roc_for_combinations(X_train, y_train, X_tests, y_tests, feature_names_combinations,
                                           classifier=RandomForestClassifier(random_state=42)):
    """
    修改后的函数，为每个特征组合计算并绘制ROC曲线，每个组合类别中的不同组合使用不同颜色。
    """
    # 定义颜色列表供不同组合使用
    colors = ['yellow', 'green', 'blue', 'red']
    if len(feature_names_combinations) == 4:
        colors = ['yellow','green', 'blue', 'red']
    elif len(feature_names_combinations) == 3:
        colors = ['green', 'blue', 'red']
    elif len(feature_names_combinations) == 2:
        colors = ['blue', 'red']

    fig, ax = plt.subplots(figsize=(10, 8))  # 创建单个图表用于绘制所有ROC曲线
    filename = ''
    for idx, comb in enumerate(feature_names_combinations, start=1):
        X_train_comb = X_train[:, [feature_names.index(f) for f in comb]]
        if X_tests:
            X_test_comb = X_tests[idx - 1][:, [feature_names.index(f) for f in comb]]
        else:
            X_test_comb = X_test[:, [feature_names.index(f) for f in comb]]

        classifier.fit(X_train_comb, y_train)
        y_pred_proba = classifier.predict_proba(X_test_comb)[:, 1]

        fpr, tpr, _ = roc_curve(y_tests[idx - 1] if y_tests else y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        color = colors[idx-1]
        if len(comb) == 5 :
            filename = 'FISUM'
        elif len(comb) == 4:
            filename = 'FOSUM'
        elif len(comb) == 3:
            filename = 'NEV_signatures'
        elif len(comb) == 2:
            filename = 'CAE+CA199'
        else:
            filename = comb[0]
        ax.plot(fpr, tpr, color=color, linewidth=4,label=f'{filename} (AUC={roc_auc:.2f})')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='lower'))
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tick_params(labelsize=30)
    ax.set_xlabel('False Positive Rate', fontdict={'size':35,'family': 'Times New Roman'})
    ax.set_ylabel('True Positive Rate', fontdict={'size':35,'family': 'Times New Roman'})
    ax.set_title('ROC curves for different feature combinations', fontdict={'size':30,'family': 'Times New Roman'})
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"ROC_for_{filename}.png")
    time.sleep(0.5)
    plt.show()
def evaluate_model(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Sensitivity (Recall):', recall_score(y_test, y_pred))
    print('Specificity:', specificity(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('F1 Score:', f1_score(y_test, y_pred))
    print('-----------------------------\n')

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    return TN / (TN + FP)

# Main execution
def ROC_curve_main(X_train, y_train, X_tests, y_tests):
    # 组合1四个曲线
    combinations_to_plot1 = [
        ['CD66b/CD63+EVs'],
        ['miR-223-3p'],
        ['miR-425-5p'],
        ['CD66b/CD63+EVs', 'miR-223-3p', 'miR-425-5p']
    ]
    # 组合2三个曲线
    combinations_to_plot2 = [
        ['CEA'],
        ['CA199'],
        ['CEA', 'CA199']
    ]
    # 组合3两个曲线
    combinations_to_plot3 = [
        ['CD66b/CD63+EVs', 'miR-223-3p', 'CEA', 'CA199'],
        feature_names
    ]

    # 绘制组合1
    evaluate_and_plot_roc_for_combinations(X_train, y_train, X_tests, y_tests, combinations_to_plot1,
                                           RandomForestClassifier(random_state=42))
    # 绘制组合2
    evaluate_and_plot_roc_for_combinations(X_train, y_train, X_tests, y_tests, combinations_to_plot2,
                                           RandomForestClassifier(random_state=42))
    # 绘制组合3
    evaluate_and_plot_roc_for_combinations(X_train, y_train, X_tests, y_tests, combinations_to_plot3,
                                           RandomForestClassifier(random_state=42))


def plot_predicted_feature_box_plots(X, y_pred, feature_names, single_feature=False):
    """
    针对每个特征或单个特征，根据预测类别绘制箱线图。
    """
    if single_feature:
        # 确保当single_feature为True时，X为一维，y_pred为一维，feature_names为单个字符串
        assert len(X.shape) == 1, "X must be a 1D array when single_feature is True."
        assert isinstance(feature_names, str), "feature_names must be a single string when single_feature is True."
        X = X.reshape(-1, 1)  # 将一维特征向量转换为二维列向量
        feature_names = [feature_names]  # 将单个特征名包装为列表以兼容后续循环

    y_pred_swapped = np.where(y_pred == 0, 1, 0)  # 假定0类别映射到'GC'，1类别映射到'HC'
    sort_indices = np.argsort(y_pred_swapped)  # 排序索引
    X_sorted = X[sort_indices]  # 根据预测值排序后的特征矩阵
    y_pred_swapped_sorted = y_pred_swapped[sort_indices]  # 排序后的预测值

    class_labels_swapped = ['HC', 'GC']

    colors = ['yellow', 'green', 'blue', 'red']
    color_cycle = cycle(colors)

    for idx, feature in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(8, 6))

        bp = plt.boxplot([X_sorted[y_pred_swapped_sorted == 1, idx], X_sorted[y_pred_swapped_sorted == 0, idx]],
                         patch_artist=True, showmeans=False, labels=class_labels_swapped)

        for patch, color in zip(bp['boxes'], color_cycle):
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
        for median_line in bp['medians']:
            median_line.set_color(next(color_cycle))

        ax.set_title(f'{feature} Box Plot Based on Predictions',
                     fontdict={'size': 25, 'family': 'Times New Roman'})
        ax.set_ylabel('Feature Value',
                      fontdict={'size': 25, 'family': 'Times New Roman'}, labelpad=0)
        ax.set_xticklabels(class_labels_swapped)
        ax.yaxis.grid(True)
        plt.tight_layout()

        # 保存图片
        safe_name = safe_filename(feature)
        plt.savefig(f'{safe_name}_predicted_boxplot.png')

        plt.show()

def box_curve_main(df, y_final, feature_names):
    X = df[feature_names].values
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.3, random_state=42)

    # 使用所有特征的模型训练与评估
    # rf_all_features = RandomForestClassifier(random_state=42)
    # evaluate_model(rf_all_features, X_train, y_train, X_test, y_test)
    # plot_predicted_feature_box_plots(X_test, rf_all_features.predict(X_test), feature_names, single_feature=False)

    # 对每个特征单独进行模型训练、评估与箱线图绘制
    for feature in feature_names:
        # 特征筛选
        single_feature_X = X[:, feature_names.index(feature)].reshape(-1, 1)

        # 划分单特征的训练集和测试集
        X_train_single, X_test_single, _, y_test_single = train_test_split(
            single_feature_X, y_final, test_size=0.3, random_state=42)

        # 为单个特征训练模型
        rf_single_feature = RandomForestClassifier(random_state=42)

        # 评估单个特征模型
        y_pred_single = rf_single_feature.fit(X_train_single, y_train).predict(X_test_single)
        print(f"{feature}预测性能的结果如下:\n")
        evaluate_model(rf_single_feature, X_train_single, y_train, X_test_single, y_test_single)

        # 绘制单个特征的箱线图
        plot_predicted_feature_box_plots(X_test_single.reshape(-1), y_pred_single, feature, single_feature=True)
if __name__ == '__main__':

    combinations_to_plot = [
        ['CD66b/CD63+EVs', 'miR-223-3p', 'miR-425-5p'],
        ['CEA', 'CA199'],
        ['CD66b/CD63+EVs', 'miR-223-3p', 'CEA', 'CA199'],
        feature_names
    ]

    # 随机森林训练
    rf_all_features = RandomForestClassifier(random_state=42)
    # evaluate_model(rf_all_features, X_train, y_train, X_test, y_test)
    X_tests = [X_test for _ in combinations_to_plot]
    y_tests = [y_test for _ in combinations_to_plot]

    # 绘制ROC曲线
    ROC_curve_main(X_train, y_train, X_tests, y_tests)

    # 绘制5个箱线图
    box_curve_main(df, y_final, feature_names=['CD66b/CD63+EVs', 'miR-223-3p', 'miR-425-5p', 'CEA', 'CA199'])

    # 评估联合特征预测效果在预测值上的效果
    for idx, comb in enumerate(combinations_to_plot, start=1):
        X_comb = df[comb].values
        X_train_comb, X_test_comb, _, y_test_comb = train_test_split(X_comb, y_final, test_size=0.3, random_state=42)

        rf_comb = RandomForestClassifier(random_state=42)
        y_pred_comb = rf_comb.fit(X_train_comb, y_train).predict(X_test_comb)

        print(f'\nResults for combination {comb}:')
        evaluate_model(rf_comb, X_train_comb, y_train, X_test_comb, y_test_comb)

        time.sleep(0.5)
