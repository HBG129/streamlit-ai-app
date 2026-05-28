import sqlite3

def generate_database():
    # 连接到数据库（如果不存在会自动创建）
    conn = sqlite3.connect('company_data.db')
    c = conn.cursor()

    # 先清理可能存在的旧表，确保写入全新数据
    c.execute('DROP TABLE IF EXISTS employees')
    c.execute('DROP TABLE IF EXISTS product_sales')

    # 1. 创建员工表
    c.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT, 
            department TEXT, 
            salary INTEGER, 
            join_date DATE
        )
    ''')

    # 2. 创建销量表
    c.execute('''
        CREATE TABLE product_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            product_name TEXT, 
            category TEXT, 
            revenue INTEGER, 
            units_sold INTEGER
        )
    ''')

    # ==========================================
    # 插入丰富的测试数据
    # ==========================================
    
    # 员工数据（模拟20人规模的公司，包含各种部门和薪资层级）
    employees_data = [
        ('张伟', '技术部', 35000, '2020-03-15'),
        ('王芳', 'HR', 15000, '2021-06-01'),
        ('李娜', '财务部', 18000, '2019-11-20'),
        ('赵强', '技术部', 28000, '2022-02-10'),
        ('刘洋', '销售部', 12000, '2023-05-18'),
        ('陈明', '销售部', 22000, '2021-08-08'),
        ('杨杰', '市场部', 25000, '2020-12-01'),
        ('黄勇', '技术部', 45000, '2018-07-15'),  # 研发大佬
        ('吴婷', '市场部', 16000, '2022-09-23'),
        ('周健', '技术部', 26000, '2023-01-05'),
        ('徐静', 'HR', 13000, '2023-10-12'),
        ('孙亮', '销售部', 30000, '2019-04-16'),  # 销冠
        ('马超', '运营部', 19000, '2021-03-25'),
        ('朱丽', '财务部', 21000, '2020-08-30'),
        ('胡斌', '技术部', 22000, '2023-07-01'),
        ('林心', '运营部', 17000, '2022-05-14'),
        ('郭峰', '市场部', 28000, '2019-01-10'),
        ('何平', '技术部', 32000, '2021-11-11'),
        ('高飞', '销售部', 14000, '2023-12-01'),
        ('郑洁', '行政部', 10000, '2022-04-20')
    ]
    
    c.executemany(
        "INSERT INTO employees (name, department, salary, join_date) VALUES (?, ?, ?, ?)",
        employees_data
    )

    # 产品销售数据（模拟不同品类的爆款与滞销品）
    sales_data = [
        ('旗舰手机 Alpha', '智能设备', 1500000, 300),
        ('降噪蓝牙耳机 Pro', '智能设备', 450000, 1500),
        ('智能手表 2代', '智能设备', 320000, 800),
        ('企业级云盘服务', '软件服务', 800000, 40),
        ('在线协作文档(年费)', '软件服务', 250000, 500),
        ('人体工学办公椅', '办公家具', 180000, 120),
        ('升降办公桌', '办公家具', 240000, 80),
        ('机械键盘 极客版', '外设配件', 90000, 300),
        ('无线静音鼠标', '外设配件', 45000, 500),
        ('便携式显示器', '外设配件', 120000, 100),
        ('桌面加湿器', '生活周边', 15000, 300),
        ('公司文化衫(定制)', '生活周边', 8000, 200),
        ('VIP专属客服包', '软件服务', 150000, 15),
        ('平板电脑 Pad Air', '智能设备', 600000, 200),
        ('护眼台灯', '生活周边', 25000, 150)
    ]

    c.executemany(
        "INSERT INTO product_sales (product_name, category, revenue, units_sold) VALUES (?, ?, ?, ?)",
        sales_data
    )

    conn.commit()
    conn.close()
    print("✅ 成功！company_data.db 数据库文件已生成，包含 20 名员工和 15 款产品数据。")

if __name__ == "__main__":
    generate_database()