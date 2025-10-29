"""
Script để test CP solver trên tất cả 56 datasets
Kết quả được lưu vào thư mục result/ với cấu trúc:
- result/images/: Chứa các file PNG trực quan hóa
- result/text/: Chứa các file TXT báo cáo
"""

import sys
from pathlib import Path
import time
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không hiển thị GUI
import matplotlib.pyplot as plt

# Import solver
from cp_vrptw_solver import CPVRPTWSolver


def test_all_datasets():
    """Test CP solver trên tất cả 56 datasets"""
    
    # Đường dẫn đến thư mục dataset - Sửa đường dẫn cho đúng
    current_dir = Path(__file__).parent  # algorithm/CP/
    code_dir = current_dir.parent.parent  # code/
    dataset_base = code_dir / "dataset"
    
    print(f"📁 Thư mục hiện tại: {current_dir}")
    print(f"📁 Thư mục code: {code_dir}")
    print(f"📁 Thư mục dataset: {dataset_base}")
    print(f"📁 Dataset tồn tại: {dataset_base.exists()}")
    print()
    
    # Danh sách các categories và files
    categories = {
        'C1': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109'],
        'C2': ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208'],
        'R1': ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112'],
        'R2': ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211'],
        'RC1': ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108'],
        'RC2': ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
    }
    
    # Tạo thư mục kết quả
    result_dir = Path(__file__).parent / 'result'
    images_dir = result_dir / 'images'
    text_dir = result_dir / 'text'
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Thống kê
    total_datasets = sum(len(files) for files in categories.values())
    success_count = 0
    failed_count = 0
    results_summary = []
    
    print("="*100)
    print("TEST CP SOLVER TRÊN TẤT CẢ 56 DATASETS")
    print("="*100)
    print(f"Thư mục kết quả: {result_dir}")
    print(f"  - Images: {images_dir}")
    print(f"  - Text: {text_dir}")
    print("="*100)
    print(f"✓ CP có thể giải TOÀN BỘ dataset (100+ khách hàng)")
    print(f"   Thời gian giải: 1 phút/dataset")
    print(f"   Sử dụng Google OR-Tools")
    print("="*100)
    
    start_time_all = time.time()
    dataset_count = 0
    
    # Duyệt qua từng category
    for category, files in categories.items():
        print(f"\n{'#'*100}")
        print(f"CATEGORY: {category} ({len(files)} datasets)")
        print(f"{'#'*100}")
        
        for dataset_name in files:
            dataset_count += 1
            dataset_path = dataset_base / category / f"{dataset_name}.csv"
            
            print(f"\n[{dataset_count}/{total_datasets}] Testing: {dataset_name}")
            print("-"*100)
            
            try:
                # Tạo solver
                solver = CPVRPTWSolver(
                    dataset_path=str(dataset_path),
                    vehicle_capacity=200,
                    max_vehicles=25
                )
                
                # Xây dựng model
                solver.build_model()
                
                # Giải bài toán (1 phút)
                result = solver.solve(time_limit=60)
                
                if result:
                    # Lưu kết quả
                    solver.visualize_solution(save=True)
                    solver.save_solution(solve_time=result['time'], status=result['status'])
                    
                    success_count += 1
                    results_summary.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Status': result['status'],
                        'Vehicles': len(result['routes']),
                        'Distance': result['objective'],
                        'Time': result['time']
                    })
                    
                    print(f"✓ Thành công: {dataset_name}")
                    print(f"  - Trạng thái: {result['status']}")
                    print(f"  - Số xe: {len(result['routes'])}")
                    print(f"  - Quãng đường: {result['objective']:.2f}")
                    print(f"  - Thời gian: {result['time']:.2f}s")
                else:
                    failed_count += 1
                    results_summary.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Status': 'Failed',
                        'Vehicles': 0,
                        'Distance': 0,
                        'Time': 0
                    })
                    print(f"✗ Thất bại: {dataset_name}")
                
            except Exception as e:
                failed_count += 1
                results_summary.append({
                    'Dataset': dataset_name,
                    'Category': category,
                    'Status': f'Error: {str(e)}',
                    'Vehicles': 0,
                    'Distance': 0,
                    'Time': 0
                })
                print(f"✗ Lỗi: {dataset_name} - {str(e)}")
    
    total_time = time.time() - start_time_all
    
    # In tóm tắt
    print("\n" + "="*100)
    print("TÓM TẮT KẾT QUẢ TEST")
    print("="*100)
    print(f"Tổng số datasets: {total_datasets}")
    print(f"Thành công: {success_count} ({success_count/total_datasets*100:.1f}%)")
    print(f"Thất bại: {failed_count} ({failed_count/total_datasets*100:.1f}%)")
    print(f"Tổng thời gian: {total_time:.2f}s ({total_time/60:.2f} phút)")
    print(f"Thời gian trung bình/dataset: {total_time/total_datasets:.2f}s")
    print("="*100)
    
    # Lưu tóm tắt vào file
    summary_df = pd.DataFrame(results_summary)
    summary_path = result_dir / 'test_summary.csv'
    summary_df.to_csv(str(summary_path), index=False, encoding='utf-8')
    print(f"\n✓ Đã lưu tóm tắt: {summary_path}")
    
    # Tạo báo cáo chi tiết
    report_path = result_dir / 'test_report.txt'
    with open(str(report_path), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BÁO CÁO TEST CP SOLVER TRÊN 56 DATASETS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Ngày giờ test: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số datasets: {total_datasets}\n")
        f.write(f"Thành công: {success_count} ({success_count/total_datasets*100:.1f}%)\n")
        f.write(f"Thất bại: {failed_count} ({failed_count/total_datasets*100:.1f}%)\n")
        f.write(f"Tổng thời gian: {total_time:.2f}s ({total_time/60:.2f} phút)\n")
        f.write(f"Thời gian trung bình: {total_time/total_datasets:.2f}s/dataset\n\n")
        
        f.write("="*100 + "\n")
        f.write("CHI TIẾT KẾT QUẢ THEO CATEGORY\n")
        f.write("="*100 + "\n\n")
        
        for category in categories.keys():
            cat_results = summary_df[summary_df['Category'] == category]
            f.write(f"\n{category}:\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Dataset':<15} {'Status':<30} {'Vehicles':<10} {'Distance':<15} {'Time (s)':<10}\n")
            f.write("-"*100 + "\n")
            
            for _, row in cat_results.iterrows():
                f.write(f"{row['Dataset']:<15} {row['Status']:<30} {row['Vehicles']:<10} "
                       f"{row['Distance']:<15.2f} {row['Time']:<10.2f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("KẾT THÚC BÁO CÁO\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Đã lưu báo cáo chi tiết: {report_path}")
    
    print("\n" + "="*100)
    print("HOÀN THÀNH TEST TẤT CẢ DATASETS!")
    print("="*100)
    print(f"\nKết quả được lưu tại: {result_dir}")
    print(f"  📊 Tóm tắt CSV: test_summary.csv")
    print(f"  📄 Báo cáo TXT: test_report.txt")
    print(f"  🖼️ Hình ảnh: images/ ({success_count} files)")
    print(f"  📝 Chi tiết: text/ ({success_count} files)")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test CP solver với nhiều datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test tất cả datasets
  python test_all_datasets.py
  
  # Test chỉ category C1
  python test_all_datasets.py --category C1
  
  # Test với giới hạn
  python test_all_datasets.py --limit 5
  
  # Tùy chỉnh thời gian
  python test_all_datasets.py --category C1 --time-limit 120
        """
    )
    
    parser.add_argument('--category', '-c', nargs='+',
                        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
                        help='Chỉ test các category cụ thể')
    
    parser.add_argument('--limit', '-l', type=int,
                        help='Giới hạn số lượng datasets')
    
    parser.add_argument('--capacity', type=int, default=200,
                        help='Sức chứa xe (mặc định: 200)')
    
    parser.add_argument('--max-vehicles', type=int, default=25,
                        help='Số xe tối đa (mặc định: 25)')
    
    parser.add_argument('--time-limit', type=int, default=60,
                        help='Giới hạn thời gian mỗi dataset (mặc định: 60s)')
    
    return parser.parse_args()


def test_with_params(categories_filter=None, limit=None, capacity=200, max_vehicles=25, time_limit=60):
    """Test với các tham số tùy chỉnh"""
    
    from cp_vrptw_solver import CPVRPTWSolver
    
    current_dir = Path(__file__).parent
    code_dir = current_dir.parent.parent
    dataset_base = code_dir / "dataset"
    
    all_categories = {
        'C1': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109'],
        'C2': ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208'],
        'R1': ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112'],
        'R2': ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211'],
        'RC1': ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108'],
        'RC2': ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
    }
    
    categories = {k: v for k, v in all_categories.items() if not categories_filter or k in categories_filter}
    
    print("="*100)
    print("TEST CONSTRAINT PROGRAMMING SOLVER")
    print("="*100)
    print(f"Categories: {list(categories.keys())}")
    if limit:
        print(f"Giới hạn: {limit} datasets")
    print(f"Thời gian giới hạn/dataset: {time_limit}s")
    print("="*100 + "\n")
    
    result_dir = current_dir / 'result'
    result_dir.mkdir(exist_ok=True)
    
    results = []
    count = 0
    
    for category, files in categories.items():
        print(f"\n{'#'*100}")
        print(f"CATEGORY: {category}")
        print(f"{'#'*100}")
        
        for dataset_name in files:
            if limit and count >= limit:
                break
                
            dataset_path = dataset_base / category / f"{dataset_name}.csv"
            if not dataset_path.exists():
                continue
            
            count += 1
            print(f"\n[{count}] {dataset_name}...", end=" ")
            
            try:
                solver = CPVRPTWSolver(str(dataset_path), capacity, max_vehicles)
                solver.build_model()
                result = solver.solve(time_limit=time_limit)
                
                if result:
                    solver.visualize_solution(save=True)
                    solver.save_solution(solve_time=result['time'], status=result['status'])
                    
                    num_vehicles = len(solver.solution)
                    total_distance = sum([r['distance'] for r in solver.solution.values()])
                    
                    results.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Vehicles': num_vehicles,
                        'Distance': round(total_distance, 2),
                        'Time': round(result['time'], 2),
                        'Status': 'Success'
                    })
                    print(f"✓ {num_vehicles} xe, {total_distance:.2f} km")
                else:
                    results.append({'Dataset': dataset_name, 'Status': 'Failed'})
                    print("❌")
            except Exception as e:
                results.append({'Dataset': dataset_name, 'Status': f'Error: {str(e)}'})
                print(f"❌ {str(e)}")
        
        if limit and count >= limit:
            break
    
    # Lưu tổng kết
    df = pd.DataFrame(results)
    df.to_csv(result_dir / 'test_summary.csv', index=False, encoding='utf-8-sig')
    
    success = len([r for r in results if r['Status'] == 'Success'])
    print(f"\n{'='*100}")
    print(f"KẾT QUẢ: {success}/{len(results)} thành công")
    print(f"{'='*100}")


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.category or args.limit:
        test_with_params(args.category, args.limit, args.capacity, args.max_vehicles, args.time_limit)
    else:
        test_all_datasets()
