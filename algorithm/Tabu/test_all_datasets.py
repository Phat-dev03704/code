"""
Test Tabu Search Algorithm trên tất cả 56 datasets Solomon VRPTW
"""

import sys
from pathlib import Path
import pandas as pd
import time
import matplotlib
import argparse
matplotlib.use('Agg')  # Sử dụng backend không cần GUI

# Import solver
from tabu_vrptw_solver import TabuSearchVRPTWSolver


def test_all_datasets():
    """Test Tabu Search solver trên tất cả 56 datasets"""
    
    print("="*100)
    print("TEST TABU SEARCH ALGORITHM TRÊN TẤT CẢ DATASETS")
    print("="*100)
    print("Thuật toán: Tabu Search (Metaheuristic)")
    print("Đặc điểm: Có cơ chế memory (Tabu List), thoát khỏi local optimum")
    print("Ưu điểm: Chất lượng solution rất cao, tốt hơn Local Search thông thường")
    print("⭐ SIÊU THAM SỐ: Tự động tính toán tối ưu cho từng dataset")
    print("Tốc độ dự kiến: 20-60 giây/dataset")
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
            solver = TabuSearchVRPTWSolver(
                dataset_path=str(dataset_info['path']),
                vehicle_capacity=200,
                max_vehicles=25
            )
            
            # ⭐ TỰ ĐỘNG LẤY SIÊU THAM SỐ TỐI ƯU CHO TỪNG DATASET
            hyperparams = solver.get_recommended_hyperparameters()
            
            print(f"  📊 Siêu tham số tự động:")
            print(f"     - Tabu tenure: {hyperparams['tabu_tenure']}")
            print(f"     - Max iterations: {hyperparams['max_iterations']}")
            print(f"     - Adaptive: {hyperparams['adaptive_tabu_tenure']}")
            
            # Giải bài toán với siêu tham số tối ưu
            result = solver.solve(
                time_limit=hyperparams['time_limit'],
                max_iterations=hyperparams['max_iterations'],
                tabu_tenure=hyperparams['tabu_tenure'],
                diversification_threshold=hyperparams['diversification_threshold'],
                aspiration_plus=hyperparams['aspiration_plus'],
                adaptive_tabu_tenure=hyperparams['adaptive_tabu_tenure'],
                neighborhood_size_limit=hyperparams['neighborhood_size_limit']
            )
            
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
        f.write("BÁO CÁO TEST TABU SEARCH ALGORITHM TRÊN TẤT CẢ DATASETS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Thời gian test: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số datasets: {len(all_datasets)}\n")
        f.write(f"Thuật toán: Tabu Search (Metaheuristic)\n")
        f.write(f"⭐ Siêu tham số: Tự động tối ưu cho từng dataset\n")
        f.write(f"Time limit: 60 giây/dataset\n")
        f.write(f"Adaptive tabu tenure: Bật cho dataset >50 KH\n")
        f.write(f"Aspiration criterion: Bật\n\n")
        
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
        description='Test Tabu Search với datasets Solomon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test tất cả datasets
  python test_all_datasets.py
  
  # Test category C1
  python test_all_datasets.py --category C1
  
  # Test 3 datasets đầu tiên
  python test_all_datasets.py --limit 3
  
  # Test với Tabu parameters tùy chỉnh
  python test_all_datasets.py --category R1 --tabu-tenure 20 --max-iter 2000
        """
    )
    
    parser.add_argument('--category', type=str,
                        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
                        help='Chỉ test category này')
    
    parser.add_argument('--limit', type=int,
                        help='Giới hạn số dataset test')
    
    parser.add_argument('--capacity', type=int, default=200,
                        help='Sức chứa xe (mặc định: 200)')
    
    parser.add_argument('--max-vehicles', type=int, default=25,
                        help='Số xe tối đa (mặc định: 25)')
    
    parser.add_argument('--tabu-tenure', type=int,
                        help='Kích thước Tabu List')
    
    parser.add_argument('--max-iter', type=int,
                        help='Số iterations tối đa')
    
    parser.add_argument('--no-improve-limit', type=int,
                        help='Giới hạn iterations không cải thiện')
    
    parser.add_argument('--time-limit', type=int,
                        help='Giới hạn thời gian (giây)')
    
    return parser.parse_args()


def test_with_params(category_filter=None, limit=None, **kwargs):
    """Test datasets với tham số tùy chỉnh"""
    
    print("="*80)
    print("🔍 Phương pháp: Tabu Search")
    print("="*80)
    
    tabu_params = {}
    if kwargs.get('tabu_tenure'):
        tabu_params['tabu_tenure'] = kwargs['tabu_tenure']
    if kwargs.get('max_iter'):
        tabu_params['max_iterations'] = kwargs['max_iter']
    if kwargs.get('no_improve_limit'):
        tabu_params['no_improve_limit'] = kwargs['no_improve_limit']
    if kwargs.get('time_limit'):
        tabu_params['time_limit'] = kwargs['time_limit']
    
    if tabu_params:
        print(f"Tabu Parameters: {tabu_params}")
        print("="*80)
    
    dataset_root = Path(__file__).parent.parent.parent / "dataset"
    categories = [category_filter] if category_filter else ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    
    all_results = []
    
    for category in categories:
        category_path = dataset_root / category
        if not category_path.exists():
            continue
            
        dataset_files = sorted(category_path.glob("*.csv"))
        if limit:
            dataset_files = dataset_files[:limit]
        
        for i, dataset_file in enumerate(dataset_files, 1):
            print(f"\n{'='*80}")
            print(f"📦 [{category}] {i}/{len(dataset_files)}: {dataset_file.stem}")
            print(f"{'='*80}")
            
            try:
                solver = TabuSearchVRPTWSolver(
                    dataset_path=str(dataset_file),
                    vehicle_capacity=kwargs.get('capacity', 200),
                    max_vehicles=kwargs.get('max_vehicles', 25)
                )
                
                hyperparams = solver.get_recommended_hyperparameters()
                hyperparams.update(tabu_params)
                
                result = solver.solve(**hyperparams)
                
                if result:
                    solver.visualize_solution(save=True)
                    solver.save_solution(
                        solve_time=result['time'],
                        status=result['status'],
                        iterations=result['iterations'],
                        improvements=result['improvements']
                    )
                    
                    all_results.append({
                        'Category': category,
                        'Dataset': dataset_file.stem,
                        'Vehicles': len(solver.solution),
                        'Distance': sum([r['distance'] for r in solver.solution.values()]),
                        'Time': result['time'],
                        'Iterations': result['iterations'],
                        'Improvements': result['improvements']
                    })
                    print(f"✓ Thành công!")
                else:
                    print(f"❌ Không tạo được solution")
                    
            except Exception as e:
                print(f"❌ Lỗi: {str(e)}")
    
    if all_results:
        result_dir = Path(__file__).parent / "result"
        result_dir.mkdir(exist_ok=True)
        
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(result_dir / "test_summary.csv", index=False)
        
        print(f"\n{'='*80}")
        print("📊 TỔNG KẾT:")
        print(f"{'='*80}")
        for cat in categories:
            cat_data = summary_df[summary_df['Category'] == cat]
            if len(cat_data) > 0:
                print(f"{cat}: Avg Vehicles={cat_data['Vehicles'].mean():.1f}, Avg Distance={cat_data['Distance'].mean():.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
        test_with_params(
            category_filter=args.category,
            limit=args.limit,
            capacity=args.capacity,
            max_vehicles=args.max_vehicles,
            tabu_tenure=args.tabu_tenure,
            max_iter=args.max_iter,
            no_improve_limit=args.no_improve_limit,
            time_limit=args.time_limit
        )
    else:
        test_all_datasets()
