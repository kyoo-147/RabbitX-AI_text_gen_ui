"""
Mô-đun này triển khai chức năng điểm chuẩn để đánh giá hiệu suất của đường dẫn nhúng. Nó mong đợi một tệp JSON cấu hình. Nó phải có câu hỏi và văn bản được mong đợi.
Đối với mỗi câu hỏi, điều cần thiết là phải có các biến thể của câu hỏi đó. Ngôn ngữ rất linh hoạt và mỗi người có thể có quan điểm riêng về cách họ có thể hỏi nó.

Cuối cùng, nó sẽ lưu kết quả bên trong tệp benchmark_{sysdate}.txt trong thư mục chính.

Hàm benchmark sẽ trả về điểm dưới dạng số nguyên.
"""
import datetime
import json
import os

from pathlib import Path

from .data_processor import process_and_add_to_collector, preprocess_text
from .parameters import get_chunk_count, get_max_token_count
from .utils import create_metadata_source

def benchmark(config_path, collector):
    # Get the current system date
    sysdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{sysdate}.txt"
    
    # Open the log file in append mode
    with open(filename, 'a') as log:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        total_points = 0
        max_points = 0

        for item in data:
            filepath = item["text"]
            corpus = ""

            # Check if the file exists
            if os.path.isfile(Path(filepath)):
                # Open the file and read its content
                with open(Path(filepath), 'r') as file:
                    corpus = file.read()
                process_and_add_to_collector(corpus, collector, True, create_metadata_source('benchmark'))
            else:
                raise f'Không thể tìm thấy tập tin được chỉ định {filepath}.'

            for question_group in item["questions"]:
                question_variants = question_group["question_variants"]
                criteria = question_group["criteria"]
                
                for q in question_variants:
                    max_points += len(criteria)
                    processed_text = preprocess_text(q)

                    # Get the most similar chunks
                    results = collector.get_sorted_by_dist(processed_text, n_results=get_chunk_count(), max_token_count=get_max_token_count())

                    points = 0
                    
                    for c in criteria:
                        for p in results:
                            if c in p:
                                points += 1
                                total_points += 1
                                break

                    info = f"Câu hỏi '{q}' được {points}/{len(criteria)} điểm."
                    print(info, file=log)

                print('\n---\n', file=log)

        print(f'##Tổng số điểm:\n\n{total_points}/{max_points}', file=log)

    return total_points, max_points