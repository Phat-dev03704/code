"""
Script tự động tạo trực quan hóa cho TẤT CẢ 56 datasets Solomon VRPTW
"""

from visualize_vrptw import VRPTWVisualizer
from pathlib import Path
import time
import argparse
import sys

def visualize_all_datasets():
    """Tạo trực quan hóa cho tất cả 56 datasets"""
    
    # Đường dẫn đến thư mục dataset
    dataset_root = Path("../dataset")
    
    # Các categories
    categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    
    # Đếm số lượng
    total_datasets = 0
    processed = 0
    failed = []
    
    print("="*80)
    print("BẮT ĐẦU TRỰC QUAN HÓA TẤT CẢ DATASETS SOLOMON VRPTW")
    print("="*80)
    
    # Đếm tổng số file
    for category in categories:
        category_path = dataset_root / category
        if category_path.exists():
            csv_files = list(category_path.glob("*.csv"))
            total_datasets += len(csv_files)
    
    print(f"\nTổng số datasets tìm thấy: {total_datasets}")
    print(f"Thời gian ước tính: ~{total_datasets * 0.5:.1f} phút")
    print("\n" + "="*80 + "\n")
    
    start_time = time.time()
    
    # Xử lý từng category
    for category in categories:
        category_path = dataset_root / category
        
        if not category_path.exists():
            print(f"⚠️  Bỏ qua {category} - Thư mục không tồn tại")
            continue
        
        # Lấy tất cả file CSV trong category
        csv_files = sorted(category_path.glob("*.csv"))
        
        print(f"\n{'='*80}")
        print(f"📁 CATEGORY: {category} - {len(csv_files)} datasets")
        print(f"{'='*80}\n")
        
        # Xử lý từng dataset
        for idx, dataset_file in enumerate(csv_files, 1):
            try:
                processed += 1
                dataset_name = dataset_file.stem
                
                print(f"[{processed}/{total_datasets}] Đang xử lý: {category}/{dataset_name}.csv")
                
                # Tạo visualizer
                viz = VRPTWVisualizer(str(dataset_file))
                
                # In thống kê ngắn gọn
                print(f"  ├─ Số khách hàng: {len(viz.customers)}")
                print(f"  ├─ Tổng demand: {viz.customers['DEMAND'].sum()}")
                
                # Tạo các biểu đồ và LƯU ẢNH
                print(f"  ├─ Đang tạo 5 biểu đồ...")
                viz.plot_all(save=True)
                
                print(f"  └─ ✅ Hoàn thành: {dataset_name}")
                print()
                
            except Exception as e:
                print(f"  └─ ❌ LỖI: {str(e)}")
                failed.append(f"{category}/{dataset_file.name}")
                print()
    
    # Tổng kết
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print("KẾT QUẢ TỔNG KẾT")
    print("="*80)
    print(f"✅ Đã xử lý thành công: {processed - len(failed)}/{total_datasets} datasets")
    print(f"❌ Thất bại: {len(failed)}/{total_datasets} datasets")
    print(f"⏱️  Thời gian thực hiện: {elapsed_time/60:.2f} phút")
    
    if failed:
        print(f"\n⚠️  Các file bị lỗi:")
        for f in failed:
            print(f"   - {f}")
    
    print(f"\n📁 Tất cả ảnh đã được lưu trong thư mục: visualize dataset/")
    print("="*80)
    
    # Tạo summary report
    create_summary_report(total_datasets, processed - len(failed), failed, elapsed_time)


def create_summary_report(total, success, failed_list, elapsed_time):
    """Tạo file báo cáo tổng kết"""
    
    report_path = "visualization_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BÁO CÁO TRỰC QUAN HÓA DATASETS SOLOMON VRPTW\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tổng số datasets: {total}\n")
        f.write(f"Thành công: {success}\n")
        f.write(f"Thất bại: {len(failed_list)}\n")
        f.write(f"Thời gian: {elapsed_time/60:.2f} phút\n\n")
        
        if failed_list:
            f.write("Các file bị lỗi:\n")
            for item in failed_list:
                f.write(f"  - {item}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CÁC LOẠI ẢNH ĐÃ TẠO CHO MỖI DATASET:\n")
        f.write("="*80 + "\n\n")
        f.write("1. [dataset_name]_locations.png - Vị trí khách hàng và depot\n")
        f.write("2. [dataset_name]_time_windows.png - Gantt chart cửa sổ thời gian\n")
        f.write("3. [dataset_name]_demand.png - Phân phối nhu cầu\n")
        f.write("4. [dataset_name]_tw_demand.png - Time window vs Demand\n")
        f.write("5. [dataset_name]_heatmaps.png - Phân tích không gian-thời gian\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n📄 Báo cáo chi tiết đã được lưu: {report_path}")


def visualize_by_category(category_name):
    """Trực quan hóa chỉ một category cụ thể"""
    
    dataset_root = Path("../dataset")
    category_path = dataset_root / category_name
    
    if not category_path.exists():
        print(f"❌ Thư mục {category_name} không tồn tại!")
        return
    
    csv_files = sorted(category_path.glob("*.csv"))
    print(f"\n📁 Trực quan hóa category: {category_name}")
    print(f"Số lượng datasets: {len(csv_files)}\n")
    
    for idx, dataset_file in enumerate(csv_files, 1):
        try:
            print(f"[{idx}/{len(csv_files)}] Đang xử lý: {dataset_file.name}")
            viz = VRPTWVisualizer(str(dataset_file))
            viz.plot_all(save=True)
            print(f"✅ Hoàn thành\n")
        except Exception as e:
            print(f"❌ Lỗi: {str(e)}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Trực quan hóa hàng loạt datasets Solomon VRPTW',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Trực quan hóa TẤT CẢ datasets (56 datasets)
  python visualize_all_datasets.py

  # Trực quan hóa chỉ một category
  python visualize_all_datasets.py --category C1
  python visualize_all_datasets.py --category R1
  python visualize_all_datasets.py --category RC2

  # Trực quan hóa nhiều categories
  python visualize_all_datasets.py --category C1 C2 R1

  # Chỉ xử lý một số lượng datasets giới hạn
  python visualize_all_datasets.py --limit 10

  # Xử lý category C1 với giới hạn 5 datasets
  python visualize_all_datasets.py --category C1 --limit 5

  # Chỉ định thư mục dataset khác
  python visualize_all_datasets.py --dataset-root ../data/solomon

  # Không tạo summary report
  python visualize_all_datasets.py --no-report

  # Tạo biểu đồ riêng lẻ thay vì tổng hợp
  python visualize_all_datasets.py --plot-type all_individual --category C1
        """
    )
    
    parser.add_argument('--category', '-c', type=str, nargs='+',
                        choices=['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'],
                        help='Chỉ xử lý các category cụ thể (có thể chọn nhiều). Nếu không chỉ định, sẽ xử lý tất cả.')
    
    parser.add_argument('--dataset-root', '-d', type=str, default='../dataset',
                        help='Đường dẫn đến thư mục chứa datasets (mặc định: ../dataset)')
    
    parser.add_argument('--limit', '-l', type=int,
                        help='Giới hạn số lượng datasets được xử lý (để test nhanh)')
    
    parser.add_argument('--no-report', action='store_true',
                        help='Không tạo file summary report')
    
    parser.add_argument('--plot-type', '-p', type=str,
                        choices=['summary', 'all_individual'],
                        default='summary',
                        help='''Loại biểu đồ tạo cho mỗi dataset:
  summary          - Biểu đồ tổng hợp 1 file (mặc định, nhanh hơn)
  all_individual   - Tạo 5 biểu đồ riêng lẻ (chậm hơn, nhiều file hơn)''')
    
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Tiếp tục xử lý nếu gặp lỗi (mặc định sẽ dừng)')
    
    return parser.parse_args()


def main():
    """Hàm chính với hỗ trợ CLI"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Kiểm tra thư mục dataset tồn tại
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ Lỗi: Thư mục dataset không tồn tại: {args.dataset_root}")
        sys.exit(1)
    
    # Xác định categories cần xử lý
    if args.category:
        categories = args.category
    else:
        categories = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    
    # Đếm số lượng
    total_datasets = 0
    processed = 0
    failed = []
    
    print("="*80)
    print("BẮT ĐẦU TRỰC QUAN HÓA DATASETS SOLOMON VRPTW")
    print("="*80)
    
    # Đếm tổng số file
    for category in categories:
        category_path = dataset_root / category
        if category_path.exists():
            csv_files = list(category_path.glob("*.csv"))
            total_datasets += len(csv_files)
    
    # Áp dụng giới hạn nếu có
    if args.limit:
        total_datasets = min(total_datasets, args.limit)
        print(f"\n⚠️  Chế độ giới hạn: Chỉ xử lý tối đa {args.limit} datasets")
    
    print(f"\nCategories được chọn: {', '.join(categories)}")
    print(f"Tổng số datasets sẽ xử lý: {total_datasets}")
    print(f"Loại biểu đồ: {args.plot_type}")
    print(f"Thời gian ước tính: ~{total_datasets * 0.5:.1f} phút")
    print("\n" + "="*80 + "\n")
    
    start_time = time.time()
    
    # Xử lý từng category
    for category in categories:
        category_path = dataset_root / category
        
        if not category_path.exists():
            print(f"⚠️  Bỏ qua {category} - Thư mục không tồn tại")
            continue
        
        # Lấy tất cả file CSV trong category
        csv_files = sorted(category_path.glob("*.csv"))
        
        print(f"\n{'='*80}")
        print(f"📁 CATEGORY: {category} - {len(csv_files)} datasets")
        print(f"{'='*80}\n")
        
        # Xử lý từng dataset
        for idx, dataset_file in enumerate(csv_files, 1):
            # Kiểm tra giới hạn
            if args.limit and processed >= args.limit:
                print(f"\n⚠️  Đã đạt giới hạn {args.limit} datasets. Dừng xử lý.")
                break
            
            try:
                processed += 1
                dataset_name = dataset_file.stem
                
                print(f"[{processed}/{total_datasets}] Đang xử lý: {category}/{dataset_name}.csv")
                
                # Tạo visualizer
                viz = VRPTWVisualizer(str(dataset_file))
                
                # In thống kê ngắn gọn
                print(f"  ├─ Số khách hàng: {len(viz.customers)}")
                print(f"  ├─ Tổng demand: {viz.customers['DEMAND'].sum()}")
                
                # Tạo các biểu đồ dựa trên loại
                print(f"  ├─ Đang tạo biểu đồ ({args.plot_type})...")
                
                if args.plot_type == 'summary':
                    viz.plot_comprehensive_summary(save=True)
                    viz.generate_text_report(save=True)
                elif args.plot_type == 'all_individual':
                    viz.plot_customer_locations(save=True)
                    viz.plot_time_windows(save=True)
                    viz.plot_demand_distribution(save=True)
                    viz.plot_time_window_width(save=True)
                    viz.plot_spatial_temporal_heatmap(save=True)
                
                print(f"  └─ ✅ Hoàn thành: {dataset_name}")
                print()
                
            except Exception as e:
                print(f"  └─ ❌ LỖI: {str(e)}")
                failed.append(f"{category}/{dataset_file.name}")
                print()
                
                if not args.continue_on_error:
                    print("\n❌ Dừng xử lý do gặp lỗi. Dùng --continue-on-error để tiếp tục khi có lỗi.")
                    sys.exit(1)
        
        # Thoát vòng lặp category nếu đã đạt giới hạn
        if args.limit and processed >= args.limit:
            break
    
    # Tổng kết
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print("KẾT QUẢ TỔNG KẾT")
    print("="*80)
    print(f"✅ Đã xử lý thành công: {processed - len(failed)}/{processed} datasets")
    print(f"❌ Thất bại: {len(failed)}/{processed} datasets")
    print(f"⏱️  Thời gian thực hiện: {elapsed_time/60:.2f} phút")
    
    if failed:
        print(f"\n⚠️  Các file bị lỗi:")
        for f in failed:
            print(f"   - {f}")
    
    print(f"\n📁 Tất cả ảnh đã được lưu trong thư mục: visualize dataset images/")
    print(f"📄 Tất cả báo cáo TXT đã được lưu trong thư mục: visulize dataset txt/")
    print("="*80)
    
    # Tạo summary report
    if not args.no_report:
        create_summary_report(processed, processed - len(failed), failed, elapsed_time)


if __name__ == "__main__":
    main()
