import os
import sys
import difflib

# This simulates what happens when someone modifies the original program
# but doesn't understand the global variables or initialization correctly

# Scenario 1: They might be manually calculating similarity without using the proper model
def test_scenario_1():
    print("Scenario 1: Manual calculation with wrong approach")
    
    # Import just what they think they need
    from app.services.scorer.evaluate_docx_submission import get_text_embedding
    
    # Define the test texts
    text1 = "Hội đồng tuyển sinh Trường Đại Tôn Đức Thắng năm 2022 thông báo điểm chuẩn trúng tuyển theo phương thức xét tuyển dựa trên kết quả kỳ thi tốt nghiệp THPT năm 2022 với thí sinh là học sinh trung học phổ thông thuộc khu vực 3( thí sinh khu vực 3 không có điểm ưu tiên) đủ điều kiện xét tuyển của Trường như sau:"
    text2 = "Hội đồng tuyển sinh Trường Đại học Giao Thông Vận Tải phân hiệu tại TP.HCM năm 2022 thông báo điểm chuẩn trúng tuyển theo phương thức xét tuyển dựa trên kết quả kỳ thi tốt nghiệp THPT năm 2022 với thí sinh là học sinh trung học phổ thông thuộc khu vực 3( thí sinh khu vực 3 không có điểm ưu tiên) đủ điều kiện xét tuyển của Trường như sau:"
    
    # They might try to calculate embeddings without initializing the model
    emb1 = get_text_embedding(text1)
    emb2 = get_text_embedding(text2)
    
    # This will likely return None for the embeddings
    if emb1 is None or emb2 is None:
        print("  Problem: Embeddings are None - model wasn't initialized properly")
        similarity = 0.0  # They might default to zero
    else:
        # Calculate similarity
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(emb1, emb2)
    
    print(f"  Result: {similarity}")
    return similarity


# Scenario 2: They might be overriding the global variables
def test_scenario_2():
    print("\nScenario 2: Overriding global variables incorrectly")
    
    # Import the modules but don't initialize
    import app.services.scorer.evaluate_docx_submission
    from app.services.scorer.evaluate_docx_submission import calculate_text_similarity
    
    # Incorrectly try to set the model (wrong variable names or types)
    # This won't work because they're setting local variables, not module globals
    phobert_model = None
    phobert_tokenizer = None
    device = None
    
    # Or they might try this, but with wrong variable names
    app.services.scorer.evaluate_docx_submission.model = None  # Wrong name, should be phobert_model
    app.services.scorer.evaluate_docx_submission.tokenizer = None  # Wrong name
    
    # Call the function 
    text1 = "Hội đồng tuyển sinh Trường Đại Tôn Đức Thắng năm 2022 thông báo điểm chuẩn trúng tuyển theo phương thức xét tuyển dựa trên kết quả kỳ thi tốt nghiệp THPT năm 2022 với thí sinh là học sinh trung học phổ thông thuộc khu vực 3( thí sinh khu vực 3 không có điểm ưu tiên) đủ điều kiện xét tuyển của Trường như sau:"
    text2 = "Hội đồng tuyển sinh Trường Đại học Giao Thông Vận Tải phân hiệu tại TP.HCM năm 2022 thông báo điểm chuẩn trúng tuyển theo phương thức xét tuyển dựa trên kết quả kỳ thi tốt nghiệp THPT năm 2022 với thí sinh là học sinh trung học phổ thông thuộc khu vực 3( thí sinh khu vực 3 không có điểm ưu tiên) đủ điều kiện xét tuyển của Trường như sau:"
    similarity = calculate_text_similarity(text1, text2)
    
    print(f"  Result: {similarity}")
    return similarity


def highlight_differences(text1, text2):
    differ = difflib.Differ()
    diff = list(differ.compare(text1.split(), text2.split()))
    result = []
    for token in diff:
        if token.startswith("- "):  # chỉ có trong text1
            result.append(f"[{token[2:]}](-)")  
        elif token.startswith("+ "):  # chỉ có trong text2
            result.append(f"[{token[2:]}](+)")
        elif token.startswith("  "):  # giống nhau
            result.append(token[2:])
    return " ".join(result)

# Run all the scenarios
if __name__ == "__main__":
    # Initialize the model correctly for scenario 4 to work
    print("Initializing model first...")
    from app.services.scorer.evaluate_docx_submission import init_phobert_model
    init_phobert_model()
    
    # Test all scenarios
    scenario1 = test_scenario_1()
    scenario2 = test_scenario_2()
    
    # Summary
    print("\nSummary:")
    print(f"Scenario 1 (Wrong manual calculation): {scenario1}")
    print(f"Scenario 2 (Incorrect global variables): {scenario2}")

    # print(f"Su khac nhau:/n")
    # text1 = "Trà Cổ nổi tiếng với đường bờ biển dài nhất Việt Nam - hơn 17km. Cái này là để cho lỗi, mày mà không biết thì lỏ dái."
    # text2 = "Trà Cổ nổi tiếng với đường bờ biển dài nhất Việt Nam - hơn 17km. Biển ở đây được du khách đánh giá là một trong những bãi biển Quảng Ninh đẹp nhất miền Bắc"
    # print(highlight_differences(text1, text2))