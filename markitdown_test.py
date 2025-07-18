from markitdown import MarkItDown

md = MarkItDown(enable_plugins=False) # Set to True to enable plugins
excel_result = md.convert("sales_data.xlsx")
print(f"excel_result.text_content: {excel_result.text_content}")

# 测试图片不能直接转
image_result = md.convert("病历1.png")
print(f"image_result.text_content: {image_result.text_content}")

# 图片需要模型：
"""

from markitdown import MarkItDown
from openai import OpenAI

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4o")
result = md.convert("example.jpg")
print(result.text_content)

"""



pdf_result = md.convert("depression_report_1751882054.pdf")
print(f"pdf_result.text_content: {pdf_result.text_content}")


