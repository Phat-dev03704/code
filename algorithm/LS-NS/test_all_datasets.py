"""
Test Local Search Algorithm trên tất cả 56 datasets Solomon VRPTW
"""

import sys
from pathlib import Path
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import argparse

# Import solver
from ls_vrptw_solver import LocalSearchVRPTWSolver


def test_all_datasets():
    """Test Local Search solver trên tất cả 56 datasets"""
    
    print("="*100)
    print("TEST LOCAL SEARCH / NEIGHBORHOOD SEARCH ALGORITHM TRÊN TẤT CẢ DATASETS")
    print("="*100)
    print("Thuật toán: Local Search (Improvement Heuristic)")
    print("Phép biến đổi: 2-opt, Relocate, Exchange, Cross")
    print("Ưu điểm: Cải thiện solution liên tục, chất lượng tốt")
    print("Tốc độ dự kiến: 10-30 giây/dataset (tùy số vòng lặp)")
    print("="*100)
    
    # Lấy đường dẫn đến thư mục dataset
    current_dir = Path(__file__).parent
    code_dir = current_dir.parent.parent
    dataset_dir = code_dir / "dataset"
    
    # Kiểm tra thư mục dataset có tồn tại không
    if not dataset_dir.exists():
        print(f"❌ Không tìm thấy thư mục dataset: {dataset_dir}")
        return
    
    print(f"✓ Tìm thấy thư mục dataset: {dataset_dir}\n")
    
    # Danh sách các thư mục con và files
    categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    
    # Thu thập tất cả file CSV
    all_datasets = []
    for category in categories:
        category_path = dataset_dir / category
        if category_path.exists():
            csv_files = sorted(category_path.glob('*.csv'))
            for csv_file in csv_files:
                all_datasets.append({
                    'category': category,
                    'filename': csv_file.name,
                    'path': csv_file
                })
    
    print(f"✓ Tìm thấy {len(all_datasets)} datasets để test\n")
    
    # Hiển thị danh sách
    for category in categories:
        category_datasets = [d for d in all_datasets if d['category'] == category]
        if category_datasets:
            print(f"  {category}: {len(category_datasets)} files")
    
    print("\n" + "="*100)
    print("BẮT ĐẦU TESTING")
    print("="*100 + "\n")
    
    # Danh sách kết quả
    results = []
    
    # Test từng dataset
    for idx, dataset_info in enumerate(all_datasets, 1):
        print(f"\n{'─'*100}")
        print(f"[{idx}/{len(all_datasets)}] Testing: {dataset_info['category']}/{dataset_info['filename']}")
        print(f"{'─'*100}")
        
        try:
            # Tạo solver
            solver = LocalSearchVRPTWSolver(
                dataset_path=str(dataset_info['path']),
                vehicle_capacity=200,
                max_vehicles=25
            )
            
            # Giải bài toán với time limit 60s
            result = solver.solve(time_limit=60, max_iterations=1000)
            
            # Vẽ và lưu solution
            solver.visualize_solution(save=True)
            solver.save_solution(
                solve_time=result['time'],
                status=result['status'],
                iterations=result['iterations'],
                improvements=result['improvements']
            )
            
            # Lưu kết quả
            results.append({
                'Dataset': dataset_info['filename'].replace('.csv', ''),
                'Category': dataset_info['category'],
                'Status': result['status'],
                'Vehicles': len(result['routes']),
                'Total Distance': f"{result['objective']:.2f}",
                'Solve Time (s)': f"{result['time']:.2f}",
                'Iterations': result['iterations'],
                'Improvements': result['improvements'],
                'Customers': solver.n_customers,
                'Customers Served': sum([r['num_customers'] for r in result['routes'].values()])
            })
            
            print(f"✓ Thành công!")
            print(f"  - Số xe: {len(result['routes'])}")
            print(f"  - Tổng quãng đường: {result['objective']:.2f}")
            print(f"  - Thời gian: {result['time']:.2f}s")
            print(f"  - Vòng lặp: {result['iterations']}")
            print(f"  - Số lần cải thiện: {result['improvements']}")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'Dataset': dataset_info['filename'].replace('.csv', ''),
                'Category': dataset_info['category'],
                'Status': f'Error: {str(e)[:50]}',
                'Vehicles': 'N/A',
                'Total Distance': 'N/A',
                'Solve Time (s)': 'N/A',
                'Iterations': 'N/A',
                'Improvements': 'N/A',
                'Customers': 'N/A',
                'Customers Served': 'N/A'
            })
    
    # Tạo DataFrame kết quả
    df_results = pd.DataFrame(results)
    
    # Lưu ra file CSV
    output_csv = current_dir / 'test_summary.csv'
    df_results.to_csv(str(output_csv), index=False, encoding='utf-8-sig')
    print(f"\n✓ Đã lưu tổng hợp kết quả: {output_csv}")
    
    # Tạo báo cáo chi tiết
    report_path = current_dir / 'test_report.txt'
    with open(str(report_path), 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BÁO CÁO TEST LOCAL SEARCH ALGORITHM TRÊN TẤT CẢ DATASETS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Thời gian test: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số datasets: {len(all_datasets)}\n")
        f.write(f"Thuật toán: Local Search / Neighborhood Search\n")
        f.write(f"Time limit: 60 giây/dataset\n")
        f.write(f"Max iterations: 1000\n\n")
        
        # Thống kê theo category
        f.write("="*100 + "\n")
        f.write("THỐNG KÊ THEO CATEGORY\n")
        f.write("="*100 + "\n\n")
        
        for category in categories:
            category_results = [r for r in results if r['Category'] == category]
            if category_results:
                f.write(f"\n{category}:\n")
                f.write(f"  Số datasets: {len(category_results)}\n")
                
                # Tính trung bình (chỉ với các kết quả thành công)
                successful = [r for r in category_results if 'Error' not in r['Status']]
                if successful:
                    avg_vehicles = sum([int(r['Vehicles']) for r in successful]) / len(successful)
                    avg_distance = sum([float(r['Total Distance']) for r in successful]) / len(successful)
                    avg_time = sum([float(r['Solve Time (s)']) for r in successful]) / len(successful)
                    avg_iterations = sum([int(r['Iterations']) for r in successful]) / len(successful)
                    avg_improvements = sum([int(r['Improvements']) for r in successful]) / len(successful)
                    
                    f.write(f"  Thành công: {len(successful)}/{len(category_results)}\n")
                    f.write(f"  Trung bình số xe: {avg_vehicles:.1f}\n")
                    f.write(f"  Trung bình quãng đường: {avg_distance:.2f}\n")
                    f.write(f"  Trung bình thời gian: {avg_time:.2f}s\n")
                    f.write(f"  Trung bình vòng lặp: {avg_iterations:.0f}\n")
                    f.write(f"  Trung bình số lần cải thiện: {avg_improvements:.1f}\n")
        
        # Chi tiết từng dataset
        f.write("\n" + "="*100 + "\n")
        f.write("CHI TIẾT TỪNG DATASET\n")
        f.write("="*100 + "\n\n")
        
        f.write(df_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*100 + "\n")
        f.write("KẾT THÚC BÁO CÁO\n")
        f.write("="*100 + "\n")
    
    print(f"✓ Đã lưu báo cáo chi tiết: {report_path}")
    
    # In tổng kết
    print("\n" + "="*100)
    print("TỔNG KẾT")
    print("="*100)
    
    successful_count = len([r for r in results if 'Error' not in r['Status']])
    print(f"Tổng số datasets test: {len(all_datasets)}")
    print(f"Thành công: {successful_count}")
    print(f"Thất bại: {len(all_datasets) - successful_count}")
    
    if successful_count > 0:
        successful_results = [r for r in results if 'Error' not in r['Status']]
        avg_vehicles = sum([int(r['Vehicles']) for r in successful_results]) / len(successful_results)
        avg_distance = sum([float(r['Total Distance']) for r in successful_results]) / len(successful_results)
        avg_time = sum([float(r['Solve Time (s)']) for r in successful_results]) / len(successful_results)
        avg_iterations = sum([int(r['Iterations']) for r in successful_results]) / len(successful_results)
        avg_improvements = sum([int(r['Improvements']) for r in successful_results]) / len(successful_results)
        
        print(f"\nKết quả trung bình:")
        print(f"  - Số xe: {avg_vehicles:.1f}")
        print(f"  - Quãng đường: {avg_distance:.2f}")
        print(f"  - Thời gian: {avg_time:.2f}s")
        print(f"  - Vòng lặp: {avg_iterations:.0f}")
        print(f"  - Số lần cải thiện: {avg_improvements:.1f}")
    
    print("\n" + "="*100)
    print("HOÀN THÀNH!")
    print("="*100)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test Local Search với nhiều datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test tất cả datasets
  python test_all_datasets.py
  
  # Test chỉ category C1
  python test_all_datasets.py --category C1
  
  # Test với giới hạn
  python test_all_datasets.py --limit 5
  
  # Tùy chỉnh tham số LS
  python test_all_datasets.py --category R1 --max-iter 150 --time-limit 120
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
    
    parser.add_argument('--max-iter', type=int,
                        help='Số vòng lặp tối đa (mặc định: tự động)')
    
    parser.add_argument('--no-improve-limit', type=int,
                        help='Dừng sau N vòng không cải thiện (mặc định: tự động)')
    
    parser.add_argument('--time-limit', type=int,
                        help='Giới hạn thời gian mỗi dataset (giây)')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Không lưu kết quả')
    
    return parser.parse_args()


def test_with_params(categories_filter=None, limit=None, capacity=200,
                     max_vehicles=25, max_iter=None, no_improve_limit=None,
                     time_limit=None, save_results=True):
    """Test với các tham số tùy chỉnh"""
    
    print("="*100)
    print("TEST LOCAL SEARCH / NEIGHBORHOOD SEARCH ALGORITHM")
    print("="*100)
    
    if categories_filter:
        print(f"Categories: {categories_filter}")
    else:
        print("Test tất cả 56 datasets Solomon")
    
    if limit:
        print(f"Giới hạn: {limit} datasets")
    
    print("Phép biến đổi: 2-opt, Relocate, Exchange, Cross")
    print("="*100)
    
    # Đường dẫn
    current_dir = Path(__file__).parent
    code_dir = current_dir.parent.parent
    dataset_dir = code_dir / "dataset"
    
    # Categories
    all_categories = {
        "C1": ["C101", "C102", "C103", "C104", "C105", "C106", "C107", "C108", "C109"],
        "C2": ["C201", "C202", "C203", "C204", "C205", "C206", "C207", "C208"],
        "R1": ["R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112"],
        "R2": ["R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211"],
        "RC1": ["RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108"],
        "RC2": ["RC201", "RC202", "RC203", "RC204", "RC205", "RC206", "RC207", "RC208"]
    }
    
    categories = {k: v for k, v in all_categories.items() if not categories_filter or k in categories_filter}
    
    results = []
    dataset_count = 0
    success_count = 0
    
    for category_name, datasets in categories.items():
        print(f"\n{'='*100}")
        print(f"CATEGORY: {category_name}")
        print(f"{'='*100}")
        
        for dataset_name in datasets:
            if limit and dataset_count >= limit:
                break
            
            dataset_path = dataset_dir / category_name / f"{dataset_name}.csv"
            
            if not dataset_path.exists():
                print(f"⚠️  Dataset không tồn tại: {dataset_path}")
                continue
            
            dataset_count += 1
            print(f"\n[{dataset_count}] {dataset_name}...", end=" ")
            
            try:
                solver = LocalSearchVRPTWSolver(
                    dataset_path=str(dataset_path),
                    vehicle_capacity=capacity,
                    max_vehicles=max_vehicles
                )
                
                hyperparams = solver.get_recommended_hyperparameters()
                
                # Override với tham số CLI
                if max_iter:
                    hyperparams['max_iterations'] = max_iter
                if no_improve_limit:
                    hyperparams['no_improve_limit'] = no_improve_limit
                if time_limit:
                    hyperparams['time_limit'] = time_limit
                
                result = solver.solve(**hyperparams)
                
                if result and result['status']:
                    if save_results:
                        solver.visualize_solution(save=True)
                        solver.save_solution(
                            solve_time=result['time'],
                            status=result['status'],
                            iterations=result['iterations'],
                            improvements=result['improvements']
                        )
                    
                    num_vehicles = len(solver.solution)
                    total_distance = sum([r['distance'] for r in solver.solution.values()])
                    
                    results.append({
                        'Dataset': dataset_name,
                        'Category': category_name,
                        'Vehicles': num_vehicles,
                        'Distance': round(total_distance, 2),
                        'Time': round(result['time'], 2),
                        'Iterations': result['iterations'],
                        'Improvements': result['improvements'],
                        'Status': 'Success'
                    })
                    
                    print(f"✓ {num_vehicles} xe, {total_distance:.2f} km, {result['time']:.2f}s")
                    success_count += 1
                else:
                    results.append({
                        'Dataset': dataset_name,
                        'Category': category_name,
                        'Status': 'Failed'
                    })
                    print("❌ Failed")
                    
            except Exception as e:
                results.append({
                    'Dataset': dataset_name,
                    'Category': category_name,
                    'Status': f'Error: {str(e)}'
                })
                print(f"❌ Error: {str(e)}")
        
        if limit and dataset_count >= limit:
            break
    
    # Lưu kết quả
    if save_results and results:
        output_dir = current_dir / 'result'
        output_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(results)
        summary_file = output_dir / 'test_summary.csv'
        df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ Đã lưu: {summary_file}")
    
    # Thống kê
    print(f"\n{'='*100}")
    print("TỔNG KẾT")
    print(f"{'='*100}")
    print(f"✓ Tổng số: {dataset_count} datasets")
    print(f"✓ Thành công: {success_count}")
    print(f"✓ Thất bại: {dataset_count - success_count}")
    
    if success_count > 0:
        success_results = [r for r in results if r.get('Status') == 'Success']
        avg_vehicles = sum(r['Vehicles'] for r in success_results) / len(success_results)
        avg_distance = sum(r['Distance'] for r in success_results) / len(success_results)
        avg_time = sum(r['Time'] for r in success_results) / len(success_results)
        avg_iterations = sum(r['Iterations'] for r in success_results) / len(success_results)
        avg_improvements = sum(r['Improvements'] for r in success_results) / len(success_results)
        
        print(f"\nTRUNG BÌNH:")
        print(f"  - Số xe: {avg_vehicles:.1f}")
        print(f"  - Quãng đường: {avg_distance:.2f} km")
        print(f"  - Thời gian: {avg_time:.2f}s")
        print(f"  - Vòng lặp: {avg_iterations:.0f}")
        print(f"  - Số lần cải thiện: {avg_improvements:.1f}")
    
    print(f"\n{'='*100}")
    print("HOÀN THÀNH!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.category or args.limit:
        test_with_params(
            categories_filter=args.category,
            limit=args.limit,
            capacity=args.capacity,
            max_vehicles=args.max_vehicles,
            max_iter=args.max_iter,
            no_improve_limit=args.no_improve_limit,
            time_limit=args.time_limit,
            save_results=not args.no_save
        )
    else:
        test_all_datasets()
