# Data sampling
def random_data(df ,seed, sample_size):
    # สุ่มแถวจาก DataFrame ตามจำนวนที่ระบุใน sample_size โดยใช้ค่า seed
    rng_df = df.sample(n=sample_size, random_state=seed)
    return rng_df

# For sampling observe
def Test_output_handel(df):
    i = 0

    # ทำการสุ่มข้อมูล 3 ครั้งโดยเปลี่ยนค่า seed ทีละ 1
    for i in range(3):
        random_data(df, 1+i).to_csv('rngdata{}.csv'.format(i+1), index= False) 
        print(i)

# ฟังก์ชันสำหรับเตรียมข้อมูล
def Initialize_Data(df, Y_col):
    # แยก features (X) และ target (y) จาก DataFrame
    X = df.drop(columns=[Y_col])
    y = df[Y_col]
    return X, y