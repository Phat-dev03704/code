"""
Test OR-Tools solver trên tất cả các datasets
"""

import sys
from pathlib import Path
import pandas as pd
import time
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from ortools_vrptw_solver import ORToolsVRPTWSolver


def test_all_datasets(time_limit_per_dataset=60):
    """Test trên tất cả datasets"""
    
    dataset_base = Path(__file__).parent.parent.parent / 'dataset'
    categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    all_results = []
    
    print("\n" + "="*120)
    print("TESTING OR-TOOLS (GOOGLE OPTIMIZATION) ON ALL DATASETS")
    print("="*120)
    
    total_start_time = time.time()
    
    for category in categories:
        category_path = dataset_base / category
        
        if not category_path.exists():
            print(f"\n⚠️ Không tìm thấy thư mục: {category_path}")
            continue
        
        csv_files = sorted(list(category_path.glob('*.csv')))
        
        print(f"\n{'='*120}")
        print(f"CATEGORY: {category} - {len(csv_files)} datasets")
        print(f"{'='*120}")
        
        for csv_file in csv_files:
            dataset_name = csv_file.stem
            
            print(f"\n{'─'*120}")
            print(f"Testing: {dataset_name}")
            print(f"{'─'*120}")
            
            try:
                solver = ORToolsVRPTWSolver(
                    dataset_path=str(csv_file),
                    vehicle_capacity=200,
                    max_vehicles=30
                )
                
                solution = solver.solve(time_limit=time_limit_per_dataset)
                
                output_dir = Path(__file__).parent / 'result'
                result_data = solver.save_all_results(output_dir)
                
                all_results.append(result_data)
                
                print(f"\n✓ {dataset_name}: "
                      f"Distance={solution['total_distance']:.2f}, "
                      f"Vehicles={solution['num_vehicles']}, "
                      f"Time={solver.solve_time:.2f}s")
                
            except Exception as e:
                print(f"\n❌ Lỗi khi test {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'Dataset': dataset_name,
                    'Algorithm': 'OR-Tools',
                    'Status': 'Error',
                    'Total_Distance': None,
                    'Num_Vehicles': None,
                    'Customers_Served': None,
                    'Solve_Time_s': None,
                    'Error': str(e)
                })
    
    total_time = time.time() - total_start_time
    
    # Lưu kết quả
    if all_results:
        output_dir = Path(__file__).parent / 'result'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_results = pd.DataFrame(all_results)
        csv_path = output_dir / 'ortools_summary.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*120}")
        print("TỔNG KẾT")
        print(f"{'='*120}")
        print(f"Tổng số datasets: {len(all_results)}")
        print(f"Thành công: {len([r for r in all_results if r.get('Status') != 'Error'])}")
        print(f"Lỗi: {len([r for r in all_results if r.get('Status') == 'Error'])}")
        print(f"Tổng thời gian: {total_time:.2f}s ({total_time/60:.2f} phút)")
        print(f"\n✓ Đã lưu summary: {csv_path}")
        
        df_success = df_results[df_results['Status'] != 'Error']
        
        if not df_success.empty:
            df_success['Category'] = df_success['Dataset'].str[:2]
            
            print(f"\n{'='*120}")
            print("THỐNG KÊ THEO CATEGORY")
            print(f"{'='*120}")
            
            summary = df_success.groupby('Category').agg({
                'Total_Distance': ['mean', 'min', 'max'],
                'Num_Vehicles': ['mean', 'min', 'max'],
                'Solve_Time_s': ['mean', 'min', 'max'],
                'Dataset': 'count'
            }).round(2)
            
            print(summary)
            
            print(f"\n{'='*120}")
            print("TOP 10 KẾT QUẢ TỐT NHẤT")
            print(f"{'='*120}")
            
            top10 = df_success.nsmallest(10, 'Total_Distance')[
                ['Dataset', 'Total_Distance', 'Num_Vehicles', 'Solve_Time_s']
            ]
            print(top10.to_string(index=False))
    
    print(f"\n{'='*120}")
    print("HOÀN THÀNH!")
    print(f"{'='*120}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test OR-Tools solver với nhiều datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test tất cả datasets
  python test_all_datasets.py
  
  # Test với time limit khác
  python test_all_datasets.py --time-limit 120
  
  # Test chỉ category C1
  python test_all_datasets.py --category C1
  
  # Test 5 datasets đầu
  python test_all_datasets.py --limit 5
        """
    )
    
    parser.add_argument('--category', '-c', nargs='+',
                        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
                        help='Chỉ test các category cụ thể')
    
    parser.add_argument('--limit', '-l', type=int,
                        help='Giới hạn số lượng datasets')
    
    parser.add_argument('--time-limit', type=int, default=60,
                        help='Giới hạn thời gian mỗi dataset (giây, mặc định: 60)')
    
    parser.add_argument('--capacity', type=int, default=200,
                        help='Sức chứa xe (mặc định: 200)')
    
    parser.add_argument('--max-vehicles', type=int, default=25,
                        help='Số xe tối đa (mặc định: 25)')
    
    return parser.parse_args()


def test_with_params(categories_filter=None, limit=None, time_limit=60, 
                     capacity=200, max_vehicles=25):
    """Test với các tham số tùy chỉnh"""
    
    dataset_base = Path(__file__).parent.parent.parent / 'dataset'
    all_categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    categories = categories_filter if categories_filter else all_categories
    
    all_results = []
    
    print("\n" + "="*120)
    print("TESTING OR-TOOLS (GOOGLE OPTIMIZATION) ON DATASETS")
    print("="*120)
    
    if categories_filter:
        print(f"Categories: {categories_filter}")
    if limit:
        print(f"Giới hạn: {limit} datasets")
    print(f"Time limit: {time_limit}s/dataset")
    print("="*120)
    
    total_start_time = time.time()
    dataset_count = 0
    
    for category in categories:
        category_path = dataset_base / category
        
        if not category_path.exists():
            print(f"\n⚠️ Không tìm thấy thư mục: {category_path}")
            continue
        
        csv_files = sorted(list(category_path.glob('*.csv')))
        
        print(f"\n{'='*120}")
        print(f"CATEGORY: {category} - {len(csv_files)} datasets")
        print(f"{'='*120}")
        
        for dataset_path in csv_files:
            if limit and dataset_count >= limit:
                break
            
            dataset_count += 1
            dataset_name = dataset_path.stem
            
            print(f"\n[{dataset_count}] {dataset_name}...", end=" ", flush=True)
            
            try:
                solver = ORToolsVRPTWSolver(
                    str(dataset_path), 
                    vehicle_capacity=capacity,
                    max_vehicles=max_vehicles
                )
                
                result = solver.solve(time_limit_seconds=time_limit)
                
                if result and result['status'] == 'Success':
                    solver.visualize_solution(save=True)
                    solver.save_solution()
                    
                    num_vehicles = len(solver.solution)
                    total_distance = sum([r['distance'] for r in solver.solution.values()])
                    
                    all_results.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Status': result['status'],
                        'Num_Vehicles': num_vehicles,
                        'Total_Distance': round(total_distance, 2),
                        'Solve_Time_s': round(result['time'], 2)
                    })
                    
                    print(f"✓ {num_vehicles} xe, {total_distance:.2f} km, {result['time']:.2f}s")
                else:
                    status = result['status'] if result else 'Failed'
                    all_results.append({
                        'Dataset': dataset_name,
                        'Category': category,
                        'Status': status
                    })
                    print(f"❌ {status}")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                all_results.append({
                    'Dataset': dataset_name,
                    'Category': category,
                    'Status': f'Error: {str(e)}'
                })
        
        if limit and dataset_count >= limit:
            break
    
    # Lưu kết quả
    if all_results:
        output_dir = Path(__file__).parent / 'result'
        output_dir.mkdir(exist_ok=True)
        
        df = pd.DataFrame(all_results)
        summary_file = output_dir / 'ortools_summary.csv'
        df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*120}")
        print("TỔNG KẾT")
        print(f"{'='*120}")
        
        df_success = df[df['Status'] == 'Success']
        success_count = len(df_success)
        
        print(f"✓ Tổng số: {len(all_results)} datasets")
        print(f"✓ Thành công: {success_count}")
        print(f"✓ Thất bại: {len(all_results) - success_count}")
        
        if success_count > 0:
            avg_vehicles = df_success['Num_Vehicles'].mean()
            avg_distance = df_success['Total_Distance'].mean()
            avg_time = df_success['Solve_Time_s'].mean()
            
            print(f"\nTRUNG BÌNH:")
            print(f"  - Số xe: {avg_vehicles:.1f}")
            print(f"  - Quãng đường: {avg_distance:.2f} km")
            print(f"  - Thời gian: {avg_time:.2f}s")
        
        print(f"\n✓ Đã lưu: {summary_file}")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*120}")
    print(f"Tổng thời gian: {total_time:.1f}s")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.category or args.limit:
        test_with_params(
            categories_filter=args.category,
            limit=args.limit,
            time_limit=args.time_limit,
            capacity=args.capacity,
            max_vehicles=args.max_vehicles
        )
    else:
        test_all_datasets(time_limit_per_dataset=args.time_limit)
    