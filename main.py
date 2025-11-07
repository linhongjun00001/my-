import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 初始化应用
st.set_page_config(page_title="门店选址评估模型", layout="wide")
st.title("门店选址评估模型")
st.write("通过多维度分析，帮助您评估潜在的门店位置")

# 创建标签页
tab1, tab2, tab3 = st.tabs(["单店评估", "多店对比", "数据分析"])

# 评估维度权重设置
with st.sidebar:
    st.header("评估维度权重设置")
    st.write("调整各维度在最终评分中的权重")
    
    # 人流量权重
    foot_traffic_weight = st.slider("人流量权重", 0.1, 0.5, 0.3, 0.05)
    # 租金成本权重
    rent_weight = st.slider("租金成本权重", 0.1, 0.4, 0.2, 0.05)
    # 竞争情况权重
    competition_weight = st.slider("竞争情况权重", 0.1, 0.3, 0.15, 0.05)
    # 周边配套权重
    amenities_weight = st.slider("周边配套权重", 0.1, 0.3, 0.15, 0.05)
    # 交通便利性权重
    transportation_weight = st.slider("交通便利性权重", 0.1, 0.3, 0.1, 0.05)
    # 目标客群匹配度权重
    target_match_weight = st.slider("目标客群匹配度权重", 0.1, 0.3, 0.1, 0.05)
    
    # 确保权重总和为1
    weights_sum = foot_traffic_weight + rent_weight + competition_weight + \
                 amenities_weight + transportation_weight + target_match_weight
    
    if not np.isclose(weights_sum, 1.0):
        st.warning(f"权重总和应为1，当前为{weights_sum:.2f}。系统将自动归一化。")
        # 归一化权重
        scale_factor = 1.0 / weights_sum
        foot_traffic_weight *= scale_factor
        rent_weight *= scale_factor
        competition_weight *= scale_factor
        amenities_weight *= scale_factor
        transportation_weight *= scale_factor
        target_match_weight *= scale_factor

# 单店评估标签页
with tab1:
    st.header("单店选址评估")
    
    # 创建表单
    with st.form("store_evaluation_form"):
        st.subheader("店铺基本信息")
        col1, col2 = st.columns(2)
        
        with col1:
            location_name = st.text_input("位置名称", "示例商业街")
            area_size = st.number_input("店铺面积 (平方米)", 20, 500, 100)
            rent_cost = st.number_input("月租金 (元)", 1000, 100000, 10000)
        
        with col2:
            city_level = st.selectbox("城市等级", ["一线城市", "二线城市", "三线城市", "四线及以下城市"])
            business_district = st.selectbox("商圈类型", ["核心商圈", "区域商圈", "社区商圈", "特色商圈"])
            lease_years = st.number_input("租赁年限", 1, 20, 3)
        
        st.subheader("人流量数据")
        col3, col4 = st.columns(2)
        
        with col3:
            morning_traffic = st.number_input("早高峰人流量 (人/小时)", 0, 10000, 1000)
            afternoon_traffic = st.number_input("午高峰人流量 (人/小时)", 0, 10000, 1500)
            evening_traffic = st.number_input("晚高峰人流量 (人/小时)", 0, 10000, 2000)
        
        with col4:
            weekend_traffic = st.number_input("周末平均人流量 (人/小时)", 0, 10000, 2500)
            holiday_traffic = st.number_input("节假日平均人流量 (人/小时)", 0, 15000, 3000)
            pedestrian_type = st.selectbox(
                "人流类型", 
                ["购物型", "通勤型", "旅游型", "混合型"],
                help="选择该位置主要的人流类型"
            )
        
        st.subheader("竞争情况")
        col5, col6 = st.columns(2)
        
        with col5:
            competitor_count = st.number_input("直接竞争对手数量", 0, 50, 3)
            competitor_distance = st.number_input("最近竞争对手距离 (米)", 0, 5000, 200)
        
        with col6:
            market_saturation = st.slider("市场饱和度", 0, 100, 50, help="0表示不饱和，100表示高度饱和")
            competitive_advantage = st.slider("竞争优势评估", 0, 100, 60, help="您的业务相比竞争对手的优势")
        
        st.subheader("周边环境与配套")
        col7, col8 = st.columns(2)
        
        with col7:
            transportation_score = st.slider("交通便利性", 0, 10, 7)
            parking_spots = st.number_input("附近停车位数量", 0, 500, 50)
            public_transit_count = st.number_input("附近公交/地铁站数量", 0, 20, 3)
        
        with col8:
            amenities_score = st.slider("周边配套完善度", 0, 10, 8)
            residential_density = st.slider("周边住宅密度", 0, 10, 6)
            commercial_density = st.slider("周边商业密度", 0, 10, 7)
        
        st.subheader("目标客群匹配度")
        col9, col10 = st.columns(2)
        
        with col9:
            target_demographic_match = st.slider("目标人群匹配度", 0, 10, 8)
            age_group_match = st.slider("年龄结构匹配度", 0, 10, 7)
        
        with col10:
            income_level_match = st.slider("收入水平匹配度", 0, 10, 6)
            consumer_behavior_match = st.slider("消费习惯匹配度", 0, 10, 7)
        
        # 提交按钮
        submitted = st.form_submit_button("评估选址")
    
    # 处理表单提交
    if submitted:
        # 计算各维度得分
        # 1. 人流量得分 (越高越好)
        avg_daily_traffic = (morning_traffic + afternoon_traffic + evening_traffic * 2) / 4
        foot_traffic_score = min(100, (avg_daily_traffic / 100))  # 转换为0-100分
        
        # 2. 租金成本得分 (租金与面积的比率，越低越好，转换为得分)
        rent_per_sqm = rent_cost / area_size
        # 根据城市等级设置不同的租金评分标准
        city_rent_standards = {
            "一线城市": 500,
            "二线城市": 300,
            "三线城市": 200,
            "四线及以下城市": 100
        }
        standard_rent = city_rent_standards[city_level]
        rent_score = max(0, 100 - ((rent_per_sqm - standard_rent) / standard_rent) * 100)
        rent_score = min(100, rent_score)
        
        # 3. 竞争情况得分 (竞争对手越少、距离越远、市场饱和度越低、竞争优势越高越好)
        competition_score = (
            (10 - competitor_count) * 5 +  # 竞争对手数量 (反向计分)
            min(100, competitor_distance / 10) * 0.2 +  # 最近竞争对手距离
            (100 - market_saturation) * 0.3 +  # 市场饱和度 (反向计分)
            competitive_advantage * 0.2  # 竞争优势
        )
        competition_score = min(100, competition_score)
        
        # 4. 周边配套得分
        amenities_score = (
            amenities_score * 10 +  # 周边配套完善度
            residential_density * 5 +  # 周边住宅密度
            commercial_density * 5  # 周边商业密度
        )
        
        # 5. 交通便利性得分
        transportation_score = (
            transportation_score * 7 +  # 交通便利性
            min(100, parking_spots) * 0.2 +  # 附近停车位数量
            public_transit_count * 5  # 附近公交/地铁站数量
        )
        transportation_score = min(100, transportation_score)
        
        # 6. 目标客群匹配度得分
        target_match_score = (
            target_demographic_match * 25 +
            age_group_match * 25 +
            income_level_match * 25 +
            consumer_behavior_match * 25
        )
        
        # 计算加权综合得分
        overall_score = (
            foot_traffic_score * foot_traffic_weight +
            rent_score * rent_weight +
            competition_score * competition_weight +
            amenities_score * amenities_weight +
            transportation_score * transportation_weight +
            target_match_score * target_match_weight
        )
        
        # 计算投资回报预期 (简化计算)
        estimated_monthly_revenue = avg_daily_traffic * 0.1 * 30  # 假设10%的人流量会消费，平均消费100元
        monthly_profit = estimated_monthly_revenue - rent_cost
        roi_months = max(0, (area_size * 2000) / monthly_profit) if monthly_profit > 0 else float('inf')  # 假设装修成本为2000元/平方米
        
        # 显示评估结果
        st.subheader("选址评估结果")
        
        # 显示综合得分
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("综合评分", f"{overall_score:.1f}/100")
            
            # 根据分数显示评级
            if overall_score >= 80:
                st.success("选址评级: 优秀 🎉")
            elif overall_score >= 70:
                st.info("选址评级: 良好 ✅")
            elif overall_score >= 60:
                st.warning("选址评级: 一般 ⚠️")
            else:
                st.error("选址评级: 不推荐 ❌")
        
        with col2:
            # 投资回报分析
            st.write("**投资回报分析**:")
            st.write(f"预估月收入: ¥{estimated_monthly_revenue:,.0f}")
            st.write(f"月租金成本: ¥{rent_cost:,.0f}")
            st.write(f"预估月利润: ¥{monthly_profit:,.0f}")
            if roi_months != float('inf'):
                st.write(f"预计回本周期: {roi_months:.1f} 个月")
            else:
                st.error("根据当前数据，该位置预计会亏损")
        
        # 创建雷达图展示各维度得分
        st.subheader("各维度得分")
        
        # 准备雷达图数据
        categories = ['人流量', '租金成本', '竞争情况', '周边配套', '交通便利性', '客群匹配度']
        values = [
            foot_traffic_score,
            rent_score,
            competition_score,
            amenities_score,
            transportation_score,
            target_match_score
        ]
        
        # 计算雷达图角度
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        values += values[:1]  # 闭合雷达图
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # 绘制雷达图
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # 设置雷达图标签
        plt.xticks(angles[:-1], categories)
        
        # 设置y轴范围
        ax.set_ylim(0, 100)
        
        # 添加标题
        plt.title(f"{location_name} 各维度得分雷达图", size=15, y=1.1)
        
        # 显示雷达图
        st.pyplot(fig)
        
        # 显示各维度详细得分
        st.subheader("维度详细分析")
        
        # 人流量分析
        st.write("**1. 人流量分析**")
        st.write(f"平均日人流量得分: {foot_traffic_score:.1f}/100")
        st.write(f"早高峰: {morning_traffic} 人/小时")
        st.write(f"午高峰: {afternoon_traffic} 人/小时")
        st.write(f"晚高峰: {evening_traffic} 人/小时")
        st.write(f"周末: {weekend_traffic} 人/小时")
        st.write(f"节假日: {holiday_traffic} 人/小时")
        
        # 租金成本分析
        st.write("**2. 租金成本分析**")
        st.write(f"租金成本得分: {rent_score:.1f}/100")
        st.write(f"月租金: ¥{rent_cost:,.0f}")
        st.write(f"店铺面积: {area_size} 平方米")
        st.write(f"每平米租金: ¥{rent_per_sqm:.1f}")
        
        # 生成建议
        st.subheader("选址建议")
        
        if overall_score >= 80:
            st.success("建议: 该位置非常适合开设店铺，各项指标表现优异，可以考虑尽快签约。")
        elif overall_score >= 70:
            st.info("建议: 该位置条件良好，有一定的发展潜力，可以考虑签约，但建议进一步优化运营策略。")
        elif overall_score >= 60:
            st.warning("建议: 该位置表现一般，需要谨慎考虑。建议进一步调查周边情况，评估潜在风险。")
        else:
            st.error("建议: 该位置不推荐开设店铺，建议继续寻找更合适的位置。")

# 多店对比标签页
with tab2:
    st.header("多店选址对比")
    st.write("导入多组选址数据进行对比分析")
    
    # 提供示例数据下载
    if st.button("下载示例数据模板"):
        # 创建示例数据
        example_data = {
            "位置名称": ["位置1", "位置2", "位置3"],
            "城市等级": ["一线城市", "二线城市", "三线城市"],
            "商圈类型": ["核心商圈", "区域商圈", "社区商圈"],
            "店铺面积": [100, 80, 120],
            "月租金": [15000, 8000, 6000],
            "早高峰人流量": [1200, 800, 600],
            "午高峰人流量": [1800, 1000, 800],
            "晚高峰人流量": [2500, 1500, 1000],
            "周末人流量": [3000, 2000, 1200],
            "节假日人流量": [3500, 2500, 1500],
            "竞争对手数量": [3, 2, 1],
            "最近竞争对手距离": [200, 300, 500],
            "市场饱和度": [60, 50, 40],
            "竞争优势评估": [70, 65, 60],
            "交通便利性": [8, 7, 6],
            "周边配套完善度": [9, 7, 6]
        }
        
        example_df = pd.DataFrame(example_data)
        csv = example_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="点击下载示例CSV",
            data=csv,
            file_name="选址评估示例数据.csv",
            mime="text/csv"
        )
    
    # 文件上传
    uploaded_file = st.file_uploader("上传包含多个位置数据的CSV文件", type="csv")
    
    if uploaded_file is not None:
        # 读取数据
        try:
            df = pd.read_csv(uploaded_file)
            st.success("数据上传成功！")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            # 验证数据格式
            required_columns = ["位置名称", "城市等级", "商圈类型", "店铺面积", "月租金",
                               "早高峰人流量", "午高峰人流量", "晚高峰人流量", "周末人流量",
                               "节假日人流量", "竞争对手数量", "最近竞争对手距离", "市场饱和度",
                               "竞争优势评估", "交通便利性", "周边配套完善度"]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"数据缺少必要的列: {', '.join(missing_columns)}")
            else:
                # 计算各位置的评分
                scores = []
                
                for _, row in df.iterrows():
                    # 计算各维度得分 (类似单店评估的逻辑)
                    # 1. 人流量得分
                    avg_daily_traffic = (row["早高峰人流量"] + row["午高峰人流量"] + row["晚高峰人流量"] * 2) / 4
                    foot_traffic_score = min(100, (avg_daily_traffic / 100))
                    
                    # 2. 租金成本得分
                    rent_per_sqm = row["月租金"] / row["店铺面积"]
                    city_rent_standards = {
                        "一线城市": 500,
                        "二线城市": 300,
                        "三线城市": 200,
                        "四线及以下城市": 100
                    }
                    standard_rent = city_rent_standards.get(row["城市等级"], 200)
                    rent_score = max(0, 100 - ((rent_per_sqm - standard_rent) / standard_rent) * 100)
                    rent_score = min(100, rent_score)
                    
                    # 3. 竞争情况得分
                    competition_score = (
                        (10 - row["竞争对手数量"]) * 5 +
                        min(100, row["最近竞争对手距离"] / 10) * 0.2 +
                        (100 - row["市场饱和度"]) * 0.3 +
                        row["竞争优势评估"] * 0.2
                    )
                    competition_score = min(100, competition_score)
                    
                    # 4. 周边配套得分 (简化)
                    amenities_score = row["周边配套完善度"] * 10
                    
                    # 5. 交通便利性得分
                    transportation_score = row["交通便利性"] * 10
                    
                    # 6. 目标客群匹配度得分 (假设默认为70，实际应用中应从数据中读取)
                    target_match_score = 70
                    
                    # 计算加权综合得分
                    overall_score = (
                        foot_traffic_score * foot_traffic_weight +
                        rent_score * rent_weight +
                        competition_score * competition_weight +
                        amenities_score * amenities_weight +
                        transportation_score * transportation_weight +
                        target_match_score * target_match_weight
                    )
                    
                    scores.append({
                        "位置名称": row["位置名称"],
                        "综合评分": overall_score,
                        "人流量得分": foot_traffic_score,
                        "租金成本得分": rent_score,
                        "竞争情况得分": competition_score,
                        "周边配套得分": amenities_score,
                        "交通便利性得分": transportation_score,
                        "目标客群匹配度得分": target_match_score
                    })
                
                # 创建评分结果数据框
                scores_df = pd.DataFrame(scores)
                scores_df = scores_df.sort_values("综合评分", ascending=False)
                
                # 显示评分结果
                st.subheader("选址对比结果")
                st.dataframe(scores_df.style.highlight_max(subset="综合评分", color="yellow"))
                
                # 可视化对比
                st.subheader("可视化对比")
                
                # 综合评分柱状图
                st.write("**综合评分对比**")
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(scores_df["位置名称"], scores_df["综合评分"])
                ax.set_ylim(0, 100)
                ax.set_xlabel("位置")
                ax.set_ylabel("综合评分")
                ax.set_title("各位置综合评分对比")
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # 各维度对比雷达图
                st.write("**各维度得分对比雷达图**")
                
                # 选择前3个位置进行雷达图对比
                top_locations = scores_df.head(3)
                
                # 准备雷达图数据
                categories = ['人流量', '租金成本', '竞争情况', '周边配套', '交通便利性', '客群匹配度']
                
                # 计算雷达图角度
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # 闭合雷达图
                
                # 创建雷达图
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                # 为每个位置绘制雷达图
                colors = ['blue', 'red', 'green']
                for i, (_, row) in enumerate(top_locations.iterrows()):
                    values = [
                        row["人流量得分"],
                        row["租金成本得分"],
                        row["竞争情况得分"],
                        row["周边配套得分"],
                        row["交通便利性得分"],
                        row["目标客群匹配度得分"]
                    ]
                    values += values[:1]  # 闭合雷达图
                    
                    ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=row["位置名称"])
                    ax.fill(angles, values, alpha=0.1, color=colors[i])
                
                # 设置雷达图标签
                plt.xticks(angles[:-1], categories)
                
                # 设置y轴范围
                ax.set_ylim(0, 100)
                
                # 添加图例
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                
                # 添加标题
                plt.title("各位置维度得分对比雷达图", size=15, y=1.1)
                
                # 显示雷达图
                st.pyplot(fig)
                
                # 生成对比建议
                st.subheader("选址对比建议")
                best_location = scores_df.iloc[0]["位置名称"]
                worst_location = scores_df.iloc[-1]["位置名称"]
                
                st.write(f"**推荐位置**: {best_location} (综合评分: {scores_df.iloc[0]['综合评分']:.1f}/100)")
                st.write(f"**不推荐位置**: {worst_location} (综合评分: {scores_df.iloc[-1]['综合评分']:.1f}/100)")
                
                # 分析各位置的优势和劣势
                st.write("**位置优劣势分析**:")
                for _, row in scores_df.iterrows():
                    strengths = []
                    weaknesses = []
                    
                    # 找出优势和劣势维度
                    if row["人流量得分"] > 80:
                        strengths.append("人流量充足")
                    elif row["人流量得分"] < 60:
                        weaknesses.append("人流量不足")
                    
                    if row["租金成本得分"] > 80:
                        strengths.append("租金成本合理")
                    elif row["租金成本得分"] < 60:
                        weaknesses.append("租金成本较高")
                    
                    if row["竞争情况得分"] > 80:
                        strengths.append("竞争压力小")
                    elif row["竞争情况得分"] < 60:
                        weaknesses.append("竞争压力大")
                    
                    if row["周边配套得分"] > 80:
                        strengths.append("周边配套完善")
                    elif row["周边配套得分"] < 60:
                        weaknesses.append("周边配套不足")
                    
                    if row["交通便利性得分"] > 80:
                        strengths.append("交通便利")
                    elif row["交通便利性得分"] < 60:
                        weaknesses.append("交通不便")
                    
                    # 显示分析结果
                    st.write(f"**{row['位置名称']}**:")
                    if strengths:
                        st.write(f"  优势: {', '.join(strengths)}")
                    if weaknesses:
                        st.write(f"  劣势: {', '.join(weaknesses)}")
        
        except Exception as e:
            st.error(f"数据处理出错: {str(e)}")

# 数据分析标签页
with tab3:
    st.header("选址数据分析工具")
    st.write("使用聚类分析等方法发现潜在的选址模式")
    
    # 提供示例聚类数据下载
    if st.button("下载聚类分析示例数据"):
        # 创建示例聚类数据
        np.random.seed(42)
        n_samples = 50
        
        # 生成模拟数据
        locations = [f"位置{i+1}" for i in range(n_samples)]
        foot_traffic = np.random.randint(500, 5000, n_samples)
        rent_per_sqm = np.random.uniform(100, 800, n_samples)
        competitor_count = np.random.randint(0, 10, n_samples)
        transportation_score = np.random.randint(1, 10, n_samples)
        
        cluster_data = {
            "位置名称": locations,
            "人流量": foot_traffic,
            "每平米租金": np.round(rent_per_sqm, 2),
            "竞争对手数量": competitor_count,
            "交通便利性": transportation_score
        }
        
        cluster_df = pd.DataFrame(cluster_data)
        csv = cluster_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="点击下载聚类示例CSV",
            data=csv,
            file_name="选址聚类分析示例数据.csv",
            mime="text/csv"
        )
    
    # 文件上传
    uploaded_file = st.file_uploader("上传位置数据进行聚类分析", type="csv")
    
    if uploaded_file is not None:
        # 读取数据
        try:
            df = pd.read_csv(uploaded_file)
            st.success("数据上传成功！")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            # 选择用于聚类的特征
            st.subheader("特征选择")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect(
                "选择用于聚类分析的特征",
                numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
            )
            
            if selected_features:
                # 设置聚类数量
                n_clusters = st.slider("选择聚类数量", 2, 10, 3)
                
                # 执行聚类分析
                if st.button("执行聚类分析"):
                    # 数据标准化
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df[selected_features])
                    
                    # K-means聚类
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    df["聚类"] = kmeans.fit_predict(scaled_data)
                    
                    # 显示聚类结果
                    st.subheader("聚类分析结果")
                    st.dataframe(df)
                    
                    # 可视化聚类结果
                    st.subheader("聚类可视化")
                    
                    # 如果有至少两个特征，可以绘制散点图
                    if len(selected_features) >= 2:
                        st.write("**聚类散点图**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(df[selected_features[0]], df[selected_features[1]], 
                                            c=df["聚类"], cmap='viridis')
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_title(f"基于{selected_features[0]}和{selected_features[1]}的聚类结果")
                        plt.colorbar(scatter, label="聚类")
                        st.pyplot(fig)
                    
                    # 如果有至少三个特征，可以绘制3D散点图
                    if len(selected_features) >= 3:
                        st.write("**3D聚类散点图**")
                        fig = plt.figure(figsize=(12, 8))
                        ax = fig.add_subplot(111, projection='3d')
                        scatter = ax.scatter(df[selected_features[0]], df[selected_features[1]], 
                                            df[selected_features[2]], c=df["聚类"], cmap='viridis')
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_zlabel(selected_features[2])
                        ax.set_title(f"基于{selected_features[0]}、{selected_features[1]}和{selected_features[2]}的3D聚类结果")
                        plt.colorbar(scatter, label="聚类")
                        st.pyplot(fig)
                    
                    # 分析每个聚类的特点
                    st.subheader("聚类特征分析")
                    
                    # 计算每个聚类的统计信息
                    cluster_stats = df.groupby("聚类").agg({
                        **{col: ['mean', 'std'] for col in selected_features},
                        "位置名称": 'count'
                    })
                    
                    # 重命名列
                    cluster_stats.columns = [f"{col}_{stat}" if stat != 'count' else f"{col}_数量" 
                                           for col, stat in cluster_stats.columns]
                    
                    st.dataframe(cluster_stats)
                    
                    # 为每个聚类生成建议
                    st.subheader("聚类选址建议")
                    
                    for i in range(n_clusters):
                        cluster_data = df[df["聚类"] == i]
                        avg_values = cluster_data[selected_features].mean()
                        
                        st.write(f"**聚类 {i}**: (共{len(cluster_data)}个位置)")
                        
                        # 基于特征生成简单建议
                        suggestions = []
                        
                        # 假设我们知道一些特征的含义（简化示例）
                        if "人流量" in selected_features:
                            if avg_values["人流量"] > df["人流量"].median():
                                suggestions.append("人流量优势明显，适合开设需要大量客流的店铺")
                            else:
                                suggestions.append("人流量相对较低，适合开设特定客群的精品店")
                        
                        if "每平米租金" in selected_features:
                            if avg_values["每平米租金"] > df["每平米租金"].median():
                                suggestions.append("租金成本较高，适合高毛利业态")
                            else:
                                suggestions.append("租金成本适中，经营压力较小")
                        
                        if "竞争对手数量" in selected_features:
                            if avg_values["竞争对手数量"] > df["竞争对手数量"].median():
                                suggestions.append("竞争较为激烈，需要明确差异化优势")
                            else:
                                suggestions.append("竞争压力较小，有较大市场空间")
                        
                        if "交通便利性" in selected_features:
                            if avg_values["交通便利性"] > df["交通便利性"].median():
                                suggestions.append("交通便利，有利于吸引远距离顾客")
                            else:
                                suggestions.append("交通条件一般，主要服务周边客群")
                        
                        if suggestions:
                            for suggestion in suggestions:
                                st.write(f"  - {suggestion}")
                        else:
                            st.write("  - 根据所选特征无法生成具体建议，请尝试选择更多关键特征")
                    
                    # 导出聚类结果
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="下载聚类分析结果",
                        data=csv,
                        file_name="选址聚类分析结果.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"数据处理出错: {str(e)}")

# 页面底部信息
st.markdown("---")
st.caption("© 2024 门店选址评估模型 - 基于多维度分析的选址决策工具")