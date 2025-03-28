import os
from PIL import Image, ImageDraw, ImageFont

# 设置图片的路径和参数
image_folder = r'D:\projects\ext_fuction_figures_lkan'  # 图片文件夹路径
output_image = r'D:\projects\ext_fuction_figures_llkan.jpg'  # 输出图片路径
rows = 4
cols = 43
image_width = 100  # 每张图片的宽度
image_height = 100  # 每张图片的高度
font_size = 12

# 创建一个大图的空白画布
total_width = cols * image_width
total_height = rows * image_height + (rows * font_size)  # 加上文字的高度
new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

# 加载字体
font = ImageFont.load_default()

# 拼接图片
for i in range(rows):
    for j in range(cols):
        image_name = f'loren_1_{i}_{j}.png'  # 按照指定格式生成文件名
        image_path = os.path.join(image_folder, image_name)
        
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize((image_width, image_height))  # 调整图片大小
            new_image.paste(img, (j * image_width, i * image_height + i * font_size))
            
        #     #在图片下方添加名称
        #     draw = ImageDraw.Draw(new_image)
        #     draw.text((j * image_width, (i + 1) * image_height + i * font_size), image_name, fill='black', font=font)
        # else:
        #     print(f'Image {image_name} not found.')

# 保存拼接后的图片
new_image.save(output_image)
print(f'拼接后的图片已保存为 {output_image}')
