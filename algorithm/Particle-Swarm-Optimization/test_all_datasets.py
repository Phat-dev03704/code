"""
Test Particle Swarm Optimization solver với tất cả datasets Solomon
Chạy 56 datasets, tạo tổng kết và báo cáo chi tiết
"""

import sys
from pathlib import Path
import time
import pandas as pd
import argparse

# Add parent directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from pso_vrptw_solver import ParticleSwarmVRPTWSolver

def test_all_datasets():
    print("="*80)
    print("TEST PARTICLE SWARM OPTIMIZATION VỚI TẤT CẢ DATASETS SOLOMON")
    print("="*80)
    print("📊 Tổng số datasets: 56")
    print("🐦 Phương pháp: Particle Swarm Optimization (Swarm Intelligence)")
    print("✓ Tự động điều chỉnh hyperparameters theo kích thước bài toán")
    print("="*80 + "\n")
    code_dir = current_dir.parent.parent
    dataset_base = code_dir / "dataset"
    categories = {
        "C1": ["C101", "C102", "C103", "C104", "C105", "C106", "C107", "C108", "C109"],
        "C2": ["C201", "C202", "C203", "C204", "C205", "C206", "C207", "C208"],
        "R1": ["R101", "R102", "R103", "R104", "R105", "R106", "R107", "R108", "R109", "R110", "R111", "R112"],
        "R2": ["R201", "R202", "R203", "R204", "R205", "R206", "R207", "R208", "R209", "R210", "R211"],
        "RC1": ["RC101", "RC102", "RC103", "RC104", "RC105", "RC106", "RC107", "RC108"],
        "RC2": ["RC201", "RC202", "RC203", "RC204", "RC205", "RC206", "RC207", "RC208"]
    }
    results = []
    total_start_time = time.time()
    dataset_count = 0
    for category, files in categories.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category}")
        print(f"{'='*80}")
        for file_name in files:
            dataset_count += 1
            dataset_path = dataset_base / category / f"{file_name}.csv"
            if not dataset_path.exists():
                print(f"❌ Không tìm thấy: {dataset_path}")
                continue
            print(f"\n--- Dataset {dataset_count}: {file_name} ---")
            solver = ParticleSwarmVRPTWSolver(
                dataset_path=str(dataset_path),
                vehicle_capacity=200,
                max_vehicles=25
            )
            hyperparams = solver.get_recommended_hyperparameters()
            result = solver.solve(**hyperparams)
            if result:
                # Tạo hình ảnh trực quan
                solver.visualize_solution(save=True)
                # Lưu báo cáo text
                solver.save_solution(
                    solve_time=result['time'],
                    status=result['status'],
                    iterations=result['iterations']
                )
                results.append({
                    'dataset': file_name,
                    'category': category,
                    'objective': result['objective'],
                    'time': result['time'],
                    'iterations': result['iterations'],
                    'num_routes': len(result['routes'])
                })
                print(f"✓ Hoàn thành: {len(result['routes'])} xe, {result['objective']:.2f} km")
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    print(f"TỔNG KẾT TEST PSO - Tổng thời gian: {total_time:.2f} giây")
    print("="*80)
    df = pd.DataFrame(results)
    summary_path = current_dir / "result" / "test_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"✓ Đã lưu tổng kết: {summary_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test Particle Swarm Optimization với nhiều datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Test tất cả datasets
  python test_all_datasets.py
  
  # Test chỉ category C1
  python test_all_datasets.py --category C1
  
  # Test với giới hạn
  python test_all_datasets.py --limit 5
  
  # Tùy chỉnh tham số PSO
  python test_all_datasets.py --category R1 --n-particles 50 --max-iter 200
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
    
    parser.add_argument('--n-particles', type=int,
                        help='Số particles (mặc định: tự động)')
    
    parser.add_argument('--max-iter', type=int,
                        help='Số iterations tối đa (mặc định: tự động)')
    
    parser.add_argument('--time-limit', type=int,
                        help='Giới hạn thời gian mỗi dataset (giây)')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Không lưu kết quả')
    
    return parser.parse_args()


def test_with_params(categories_filter=None, limit=None, capacity=200,
                     max_vehicles=25, n_particles=None, max_iter=None,
                     time_limit=None, save_results=True):
    """Test với các tham số tùy chỉnh"""
    
    print("="*80)
    print("TEST PARTICLE SWARM OPTIMIZATION")
    print("="*80)
    
    if categories_filter:
        print(f"Categories: {categories_filter}")
    else:
        print("Test tất cả 56 datasets Solomon")
    
    if limit:
        print(f"Giới hạn: {limit} datasets")
    
    print("🐦 Phương pháp: Particle Swarm Optimization")
    print("="*80 + "\n")
    
    code_dir = current_dir.parent.parent
    dataset_base = code_dir / "dataset"
    
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
    total_start = time.time()
    
    for category_name, datasets in categories.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category_name}")
        print(f"{'='*80}")
        
        for dataset_name in datasets:
            if limit and dataset_count >= limit:
                break
            
            dataset_path = dataset_base / category_name / f"{dataset_name}.csv"
            
            if not dataset_path.exists():
                print(f"⚠️  Dataset không tồn tại: {dataset_path}")
                continue
            
            dataset_count += 1
            print(f"\n[{dataset_count}] {dataset_name}...", end=" ")
            
            try:
                solver = ParticleSwarmVRPTWSolver(
                    dataset_path=str(dataset_path),
                    vehicle_capacity=capacity,
                    max_vehicles=max_vehicles
                )
                
                hyperparams = solver.get_recommended_hyperparameters()
                
                # Override với tham số CLI
                if n_particles:
                    hyperparams['n_particles'] = n_particles
                if max_iter:
                    hyperparams['max_iterations'] = max_iter
                if time_limit:
                    hyperparams['time_limit'] = time_limit
                
                result = solver.solve(**hyperparams)
                
                if result and result['status']:
                    if save_results:
                        solver.visualize_solution(save=True)
                        solver.save_solution(
                            solve_time=result['time'],
                            status=result['status'],
                            iterations=result['iterations']
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
    
    total_time = time.time() - total_start
    
    # Lưu kết quả
    if save_results and results:
        df = pd.DataFrame(results)
        summary_path = current_dir / "result" / "test_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
        print(f"\n✓ Đã lưu: {summary_path}")
    
    # Thống kê
    print(f"\n{'='*80}")
    print("TỔNG KẾT")
    print(f"{'='*80}")
    print(f"✓ Tổng số: {dataset_count} datasets")
    print(f"✓ Thành công: {success_count}")
    print(f"✓ Thất bại: {dataset_count - success_count}")
    print(f"✓ Tổng thời gian: {total_time:.2f}s")
    
    if success_count > 0:
        success_results = [r for r in results if r.get('Status') == 'Success']
        avg_vehicles = sum(r['Vehicles'] for r in success_results) / len(success_results)
        avg_distance = sum(r['Distance'] for r in success_results) / len(success_results)
        avg_time = sum(r['Time'] for r in success_results) / len(success_results)
        
        print(f"\nTRUNG BÌNH:")
        print(f"  - Số xe: {avg_vehicles:.1f}")
        print(f"  - Quãng đường: {avg_distance:.2f} km")
        print(f"  - Thời gian: {avg_time:.2f}s")
    
    print(f"\n{'='*80}")
    print("HOÀN THÀNH!")
    print(f"{'='*80}\n")


def main():
    args = parse_arguments()
    
    if args.category or args.limit:
        test_with_params(
            categories_filter=args.category,
            limit=args.limit,
            capacity=args.capacity,
            max_vehicles=args.max_vehicles,
            n_particles=args.n_particles,
            max_iter=args.max_iter,
            time_limit=args.time_limit,
            save_results=not args.no_save
        )
    else:
        test_all_datasets()


if __name__ == "__main__":
    main()
