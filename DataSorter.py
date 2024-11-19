import pandas as pd

# Load the CSV file
file_path = '/Users/sohamsane/Documents/Coding Projects/DualDynamicTransformer/Laptop_Price.csv'
laptop_data = pd.read_csv(file_path)

# Define the brands for each new CSV
asus_brand = 'Asus'
combined_brands = ['Acer', 'Lenovo']
hp_dell_brands = ['HP', 'Dell']
model1_full = ['Asus', 'Acer', 'Lenovo']

# Filter and save Asus laptops
#asus_laptops = laptop_data[laptop_data['Brand'] == asus_brand]
#asus_laptops.to_csv('Asus_laptops.csv', index=False)

# Filter and save Model1 Full laptops
model1_full_laptops = laptop_data[laptop_data['Brand'].isin(model1_full)]
model1_full_laptops.to_csv('Model1_Full.csv', index=False)

# Filter and save Acer and Lenovo laptops
#acer_lenovo_laptops = laptop_data[laptop_data['Brand'].isin(combined_brands)]
#acer_lenovo_laptops.to_csv('Acer_Lenovo_laptops.csv', index=False)

# Filter and save HP and Dell laptops
#hp_dell_laptops = laptop_data[laptop_data['Brand'].isin(hp_dell_brands)]
#hp_dell_laptops.to_csv('HP_Dell_laptops.csv', index=False)

print("CSV files created successfully.")
