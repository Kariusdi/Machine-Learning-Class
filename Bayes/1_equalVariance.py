import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ฟังก์ชันสำหรับคำนวณ Likelihood และปรับด้วย Prior
def likelihood_with_prior(x, mean, std_dev, prior):
    likelihood = 1.0 / (np.sqrt(2 * np.pi) * std_dev) * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return likelihood

# ฟังก์ชันสำหรับคำนวณ Posterior โดยสามารถปรับค่า Prior ได้
def posterior(x, mean0, mean1, std_dev, prior0, prior1):
    likelihood0 = likelihood_with_prior(x, mean0, std_dev, prior0)
    likelihood1 = likelihood_with_prior(x, mean1, std_dev, prior1)
    
    posterior0 = likelihood0 * prior0
    posterior1 = likelihood1 * prior1
    
    evidence = posterior0 + posterior1
    posterior0 /= evidence
    posterior1 /= evidence
    
    return posterior0, posterior1

# ฟังก์ชันสำหรับคำนวณขอบเขตการตัดสินใจ (Decision Boundary) ที่คำนึงถึง Prior
def decision_boundary(mean0, mean1, std_dev, prior0, prior1):
    return (mean0 + mean1) / 2 - (std_dev**2 / (mean1 - mean0)) * np.log(prior1 / prior0)

# สร้างกราฟสำหรับพารามิเตอร์ที่กำหนดเอง
def fixed_parameters():
    mean0, mean1 = 2, 5  # ค่าเฉลี่ยของ Class 0 และ Class 1
    std_dev = 1.5  # ความแปรปรวนของทั้งสองคลาสเท่ากัน
    prior0, prior1 = 0.8, 0.2  # ค่า Prior สำหรับ Class 0 และ Class 1 (ปรับค่าได้)

    # กำหนดช่วงของ x สำหรับการพล็อต
    x_values = np.linspace(-2, 10, 200)
    
    # คำนวณ likelihood และ posterior
    likelihood0_values = likelihood_with_prior(x_values, mean0, std_dev, prior0)
    likelihood1_values = likelihood_with_prior(x_values, mean1, std_dev, prior1)
    
    posterior_values0, posterior_values1 = posterior(x_values, mean0, mean1, std_dev, prior0, prior1)

    # กำหนดจุดตัดสินใจ
    decision_point = decision_boundary(mean0, mean1, std_dev, prior0, prior1)

    # พล็อตกราฟ
    plt.figure(figsize=(15, 5))
    
    # Likelihood plot
    plt.subplot(1, 3, 1)
    plt.plot(x_values, likelihood0_values, label=f'Likelihood Class 0 (Prior={prior0})', color='blue')
    plt.plot(x_values, likelihood1_values, label=f'Likelihood Class 1 (Prior={prior1})', color='red')
    plt.title('Likelihood (weighted by Prior)')
    plt.legend()

    # Posterior plot
    plt.subplot(1, 3, 2)
    plt.plot(x_values, posterior_values0, label=f'Posterior Class 0 (Prior={prior0})', color='green')
    plt.plot(x_values, posterior_values1, label=f'Posterior Class 1 (Prior={prior1})', color='orange')
    plt.axvline(x=decision_point, color='black', linestyle='--', label='Decision Boundary')
    plt.title('Posterior with Decision Boundary')
    plt.legend()

    # # Decision Boundary plot
    # plt.subplot(1, 3, 3)
    # plt.plot(x_values, likelihood0_values, label=f'Likelihood Class 0 (Prior={prior0})', color='blue')
    # plt.plot(x_values, likelihood1_values, label=f'Likelihood Class 1 (Prior={prior1})', color='red')
    # # plt.axvline(x=decision_point, color='black', linestyle='--', label='Decision Boundary')
    # plt.title('Decision Boundary')
    # plt.legend()

    plt.show()

fixed_parameters()
