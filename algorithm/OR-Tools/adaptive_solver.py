"""
ADAPTIVE HYPERPARAMETER TUNING FOR OR-TOOLS VRPTW SOLVER
Tự động phân tích đặc điểm dataset để chọn strategy/metaheuristic phù hợp
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from ortools_vrptw_solver import ORToolsVRPTWSolver
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


# ===== BEST KNOWN SOLUTIONS (BKS) - Solomon Benchmark =====
# Nguồn: SINTEF TOP
# https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/
BKS = {
    # C1 - Clustered, short time windows
    'C101': 828.94, 'C102': 828.94, 'C103': 828.06, 'C104': 824.78,
    'C105': 828.94, 'C106': 828.94, 'C107': 828.94, 'C108': 828.94, 'C109': 828.94,
    
    # C2 - Clustered, long time windows
    'C201': 591.56, 'C202': 591.56, 'C203': 591.17, 'C204': 590.60,
    'C205': 588.88, 'C206': 588.49, 'C207': 588.29, 'C208': 588.32,
    
    # R1 - Random, short time windows
    'R101': 1650.80, 'R102': 1486.12, 'R103': 1292.68, 'R104': 1007.31,
    'R105': 1377.11, 'R106': 1252.03, 'R107': 1104.66, 'R108': 960.88,
    'R109': 1194.73, 'R110': 1118.84, 'R111': 1096.72, 'R112': 982.14,
    
    # R2 - Random, long time windows
    'R201': 1252.37, 'R202': 1191.70, 'R203': 939.50, 'R204': 825.52,
    'R205': 994.42, 'R206': 906.14, 'R207': 890.61, 'R208': 726.82,
    'R209': 909.16, 'R210': 939.37, 'R211': 885.71,
    
    # RC1 - Random-Clustered, short time windows
    'RC101': 1696.95, 'RC102': 1554.75, 'RC103': 1261.67, 'RC104': 1135.48,
    'RC105': 1629.44, 'RC106': 1424.73, 'RC107': 1230.48, 'RC108': 1139.82,
    
    # RC2 - Random-Clustered, long time windows
    'RC201': 1406.94, 'RC202': 1365.65, 'RC203': 1049.62, 'RC204': 798.46,
    'RC205': 1297.65, 'RC206': 1146.32, 'RC207': 1061.14, 'RC208': 828.14
}


class DatasetAnalyzer:
    """
    Phân tích đặc điểm dataset để lựa chọn strategy/metaheuristic phù hợp
    """
    
    @staticmethod
    def analyze_dataset(dataset_path):
        """
        Phân tích đặc điểm chi tiết của dataset
        
        Returns:
            dict: Các chỉ số đặc trưng của dataset
        """
        df = pd.read_csv(dataset_path)
        
        # Loại bỏ depot (row 0)
        customers = df.iloc[1:].copy()
        depot = df.iloc[0]
        
        n_customers = len(customers)
        
        # 1. PHÂN TÍCH KHÔNG GIAN (Spatial Analysis)
        coords = customers[['XCOORD.', 'YCOORD.']].values
        
        # Tính độ phân tán (Coefficient of Variation)
        x_cv = np.std(coords[:, 0]) / (np.mean(coords[:, 0]) + 1e-6)
        y_cv = np.std(coords[:, 1]) / (np.mean(coords[:, 1]) + 1e-6)
        spatial_dispersion = (x_cv + y_cv) / 2
        
        # Tính độ tập trung (clustering) bằng silhouette-like metric
        try:
            dist_matrix = squareform(pdist(coords, 'euclidean'))
            median_dist = np.median(dist_matrix[dist_matrix > 0])
            clustering_score = np.sum(dist_matrix < median_dist) / (dist_matrix.size)
        except:
            clustering_score = 0.5
        
        # 2. PHÂN TÍCH TIME WINDOWS
        time_windows = customers[['READY TIME', 'DUE DATE']].values
        
        # Độ rộng time window trung bình
        tw_widths = time_windows[:, 1] - time_windows[:, 0]
        avg_tw_width = np.mean(tw_widths)
        min_tw_width = np.min(tw_widths)
        
        # Tỷ lệ time windows chặt (< 50% của max)
        max_tw = np.max(tw_widths)
        tight_tw_ratio = np.sum(tw_widths < 0.5 * max_tw) / n_customers
        
        # Độ chồng lấn time windows
        tw_overlaps = 0
        for i in range(min(len(time_windows), 100)):  # Chỉ check 100 cặp đầu để tăng tốc
            for j in range(i+1, min(len(time_windows), 100)):
                if time_windows[i][0] < time_windows[j][1] and time_windows[j][0] < time_windows[i][1]:
                    tw_overlaps += 1
        tw_overlap_ratio = tw_overlaps / (100 * 99 / 2) if len(time_windows) >= 100 else tw_overlaps / (n_customers * (n_customers - 1) / 2)
        
        # 3. PHÂN TÍCH ĐỘ KHÓ
        # Tính urgency: khoảng cách đến depot vs time window
        depot_coords = depot[['XCOORD.', 'YCOORD.']].values
        distances_to_depot = np.sqrt((coords[:, 0] - depot_coords[0])**2 + 
                                     (coords[:, 1] - depot_coords[1])**2)
        urgency_scores = distances_to_depot / (tw_widths + 1e-6)
        avg_urgency = np.mean(urgency_scores)
        
        # Tính feasibility pressure
        service_times = customers['SERVICE TIME'].values
        total_service_time = np.sum(service_times)
        max_horizon = depot['DUE DATE']
        time_pressure = (total_service_time + np.sum(distances_to_depot)) / max_horizon
        
        return {
            'n_customers': n_customers,
            'spatial_dispersion': spatial_dispersion,
            'clustering_score': clustering_score,
            'avg_tw_width': avg_tw_width,
            'min_tw_width': min_tw_width,
            'tight_tw_ratio': tight_tw_ratio,
            'tw_overlap_ratio': tw_overlap_ratio,
            'avg_urgency': avg_urgency,
            'time_pressure': time_pressure
        }
    
    @staticmethod
    def select_parameters(features, dataset_type):
        """
        Chọn strategy và metaheuristic dựa trên đặc điểm dataset
        
        Args:
            features: Dict các đặc trưng từ analyze_dataset()
            dataset_type: 'C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'
        
        Returns:
            dict: Config parameters
        """
        
        # ===== QUY TẮC CHỌN STRATEGY (CẢI THIỆN CHO R DATASETS) =====
        
        # Special Rule: Dataset R (Random) cần strategy đặc biệt
        if dataset_type in ['R1', 'R2']:
            # PATH_CHEAPEST_ARC hoạt động tốt nhất cho random với TW chặt
            strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            reason_strategy = "R dataset (random) - PATH_CHEAPEST_ARC tối ưu"
        
        # Rule 1: Nếu time windows rất chặt → PATH_MOST_CONSTRAINED_ARC
        elif features['tight_tw_ratio'] > 0.7 or features['min_tw_width'] < 20:
            strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
            reason_strategy = "Time windows rất chặt"
        
        # Rule 2: Nếu clustering cao → PATH_CHEAPEST_ARC
        elif features['clustering_score'] > 0.5:
            strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            reason_strategy = "Clustering cao"
        
        # Rule 3: Nếu phân tán cao (random) → PATH_CHEAPEST_ARC (đổi từ GLOBAL)
        elif features['spatial_dispersion'] > 0.4:
            strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            reason_strategy = "Phân tán ngẫu nhiên cao"
        
        # Rule 4: Nếu urgency cao → LOCAL_CHEAPEST_INSERTION (nhanh)
        elif features['avg_urgency'] > 1.5:
            strategy = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
            reason_strategy = "Urgency cao"
        
        # Rule 5: Default - SAVINGS
        else:
            strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
            reason_strategy = "Cân bằng (default)"
        
        
        # ===== QUY TẮC CHỌN METAHEURISTIC (CẢI THIỆN CHO R DATASETS) =====
        
        # TỐI ƯU ĐỂ ĐẠT GAP 5-10% TRONG 60S
        # Tất cả đều dùng GUIDED_LOCAL_SEARCH (mạnh nhất) + 60s cố định
        
        metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        time_limit = 60  # Cố định 60s cho tất cả
        solution_limit = 1000000
        reason_meta = f"{dataset_type} - GLS 60s + full operators"
        
        
        # ===== TIME LIMIT CỐ ĐỊNH 60S =====
        # Tối ưu bằng cách bật tất cả local search operators thay vì tăng time
        
        return {
            'first_solution_strategy': strategy,
            'local_search_metaheuristic': metaheuristic,
            'time_limit': time_limit,
            'solution_limit': solution_limit,
            'reason_strategy': reason_strategy,
            'reason_metaheuristic': reason_meta
        }


class AdaptiveORToolsSolver(ORToolsVRPTWSolver):
    """
    OR-Tools Solver với Adaptive Parameter Selection
    """
    
    def __init__(self, dataset_path, **kwargs):
        """
        Khởi tạo solver với adaptive parameter selection
        
        Args:
            dataset_path: Đường dẫn file dataset
            **kwargs: Các tham số khác ghi đè
        """
        self.dataset_path = dataset_path
        dataset_name = Path(dataset_path).stem
        
        # Nhận diện loại dataset
        if dataset_name.startswith('C1'):
            self.dataset_type = 'C1'
        elif dataset_name.startswith('C2'):
            self.dataset_type = 'C2'
        elif dataset_name.startswith('R1'):
            self.dataset_type = 'R1'
        elif dataset_name.startswith('R2'):
            self.dataset_type = 'R2'
        elif dataset_name.startswith('RC1'):
            self.dataset_type = 'RC1'
        elif dataset_name.startswith('RC2'):
            self.dataset_type = 'RC2'
        else:
            self.dataset_type = 'UNKNOWN'
        
        print(f"\n{'='*100}")
        print(f"ADAPTIVE PARAMETER SELECTION - {dataset_name}")
        print(f"{'='*100}")
        
        # Phân tích dataset
        print(f"\nPhân tích đặc điểm dataset...")
        self.features = DatasetAnalyzer.analyze_dataset(dataset_path)
        
        print(f"\nĐặc điểm dataset:")
        print(f"  • Số khách hàng: {self.features['n_customers']}")
        print(f"  • Spatial Dispersion: {self.features['spatial_dispersion']:.3f} {'(Random)' if self.features['spatial_dispersion'] > 0.4 else '(Clustered)'}")
        print(f"  • Clustering Score: {self.features['clustering_score']:.3f}")
        print(f"  • Avg TW Width: {self.features['avg_tw_width']:.1f} {'(Tight)' if self.features['avg_tw_width'] < 50 else '(Relaxed)'}")
        print(f"  • Tight TW Ratio: {self.features['tight_tw_ratio']:.2%}")
        print(f"  • Time Pressure: {self.features['time_pressure']:.3f} {'(High)' if self.features['time_pressure'] > 0.8 else '(Normal)'}")
        print(f"  • Avg Urgency: {self.features['avg_urgency']:.3f}")
        
        # Chọn parameters
        self.adaptive_config = DatasetAnalyzer.select_parameters(self.features, self.dataset_type)
        
        print(f"\n{'─'*100}")
        print(f"ADAPTIVE STRATEGY SELECTED:")
        print(f"  → Strategy: {self._get_strategy_name(self.adaptive_config['first_solution_strategy'])}")
        print(f"    Lý do: {self.adaptive_config['reason_strategy']}")
        print(f"  → Metaheuristic: {self._get_metaheuristic_name(self.adaptive_config['local_search_metaheuristic'])}")
        print(f"    Lý do: {self.adaptive_config['reason_metaheuristic']}")
        print(f"  → Time Limit: {self.adaptive_config['time_limit']}s")
        print(f"  → Solution Limit: {self.adaptive_config['solution_limit']}")
        print(f"{'─'*100}\n")
        
        # Lưu strategy và metaheuristic đã chọn
        self.selected_strategy = self.adaptive_config['first_solution_strategy']
        self.selected_metaheuristic = self.adaptive_config['local_search_metaheuristic']
        
        # Khởi tạo base solver
        kwargs.setdefault('vehicle_capacity', 200)
        kwargs.setdefault('max_vehicles', 25)
        super().__init__(dataset_path, **kwargs)
    
    def solve_adaptive(self):
        """
        Giải bài toán với adaptive parameters
        """
        print(f"Đang giải bài toán với adaptive parameters...")
        
        start_time = time.time()
        
        # Tạo data model
        data = self._create_data_model()
        
        # Tạo routing model
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,
            data['vehicle_capacities'],
            True, 'Capacity'
        )
        
        # Time window constraints
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node] + data['service_times'][from_node]
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            int(1e6), int(1e6),
            False, 'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == data['depot']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        depot_idx = data['depot']
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data['time_windows'][depot_idx][0],
                data['time_windows'][depot_idx][1]
            )
        
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i))
            )
        
        # ===== ADAPTIVE SEARCH PARAMETERS (TỐI ƯU ĐỂ ĐẠT GAP 5-10%) =====
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.adaptive_config['first_solution_strategy']
        search_parameters.local_search_metaheuristic = self.adaptive_config['local_search_metaheuristic']
        search_parameters.time_limit.FromSeconds(self.adaptive_config['time_limit'])
        search_parameters.solution_limit = self.adaptive_config['solution_limit']
        search_parameters.log_search = False
        
        # THÊM CÁC THAM SỐ TỐI ƯU ĐỂ CẢI THIỆN CHẤT LƯỢNG SOLUTION
        # 1. Tăng cường local search
        search_parameters.use_full_propagation = True  # Tìm kiếm đầy đủ hơn
        
        # 2. Guided Local Search parameters - QUAN TRỌNG!
        # Điều chỉnh penalty để tìm được solution tốt hơn
        search_parameters.guided_local_search_lambda_coefficient = 0.1  # Tăng từ mặc định
        
        # 3. Local search operators - BẬT TẤT CẢ ĐỂ TỐI ƯU
        search_parameters.local_search_operators.use_relocate = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_relocate_pair = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_light_relocate_pair = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_relocate_neighbors = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_relocate_subtrip = pywrapcp.BOOL_TRUE  # Thêm mới
        search_parameters.local_search_operators.use_exchange = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_exchange_pair = pywrapcp.BOOL_TRUE  # Thêm mới
        search_parameters.local_search_operators.use_exchange_subtrip = pywrapcp.BOOL_TRUE  # Thêm mới
        search_parameters.local_search_operators.use_cross = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_cross_exchange = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_two_opt = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_or_opt = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_lin_kernighan = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_path_lns = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_full_path_lns = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_tsp_opt = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_make_active = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_relocate_expensive_chain = pywrapcp.BOOL_TRUE
        search_parameters.local_search_operators.use_extended_swap_active = pywrapcp.BOOL_TRUE  # Thêm mới
        search_parameters.local_search_operators.use_node_pair_swap_active = pywrapcp.BOOL_TRUE  # Thêm mới
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        self.solve_time = time.time() - start_time
        
        # Lưu manager và routing để có thể extract solution ngay cả khi timeout
        self.manager = manager
        self.routing = routing
        self.data = data
        
        # QUAN TRỌNG: OR-Tools vẫn trả về solution tốt nhất hiện tại ngay cả khi timeout!
        # Chỉ cần kiểm tra xem có solution nào không
        if solution:
            self.solution = self._extract_solution(data, manager, routing, solution)
            
            if self.solution:
                # Kiểm tra routing status
                status_code = routing.status()
                if status_code == 1:  # ROUTING_SUCCESS
                    status = "Optimal/Success"
                elif status_code == 2:  # ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED
                    status = "Feasible (Local optimum not reached)"
                elif status_code == 3:  # ROUTING_FAIL
                    status = "Failed"
                elif status_code == 4:  # ROUTING_FAIL_TIMEOUT
                    status = "Timeout (Best solution kept)"
                else:
                    status = f"Status code: {status_code}"
                
                # Tính gap so với BKS
                dataset_name = Path(self.dataset_path).stem
                bks_value = BKS.get(dataset_name)
                
                if bks_value:
                    gap = ((self.solution['total_distance'] - bks_value) / bks_value) * 100
                    gap_str = f"{gap:+.2f}%"
                else:
                    gap_str = "N/A"
                    gap = None
                
                print(f"\n{'='*100}")
                print(f"KẾT QUẢ ADAPTIVE SOLVER - {dataset_name}")
                print(f"{'='*100}")
                print(f"Status: {status}")
                print(f"Số xe sử dụng: {self.solution['num_vehicles']}")
                print(f"Tổng quãng đường: {self.solution['total_distance']:.2f}")
                if bks_value:
                    print(f"Best Known Solution: {bks_value:.2f}")
                    print(f"Gap: {gap_str} {'(Excellent!)' if gap <= 1 else '(Good)' if gap <= 5 else '(Acceptable)' if gap <= 10 else '(Needs improvement)'}")
                print(f"Khách hàng phục vụ: {self.solution['customers_served']}/{self.n_customers}")
                print(f"Thời gian giải: {self.solve_time:.2f}s")
                print(f"{'='*100}\n")
                
                # Lưu thêm thông tin vào solution
                self.solution['status'] = status
                self.solution['bks'] = bks_value
                self.solution['gap'] = gap
                
                # LƯU KẾT QUẢ NGAY CẢ KHI TIMEOUT - Vì đã có solution tốt nhất hiện tại
                return self.solution
            else:
                print(f"\nKhông thể extract solution!")
                return None
        else:
            print(f"\nKhông tìm được solution trong {self.adaptive_config['time_limit']}s!")
            print(f"Có thể cần tăng time_limit hoặc thử strategy khác.")
            return None
    
    @staticmethod
    def _get_strategy_name(enum_value):
        """Chuyển enum strategy thành tên dễ đọc"""
        strategy_names = {
            0: "AUTOMATIC",
            6: "PATH_CHEAPEST_ARC",
            3: "PARALLEL_CHEAPEST_INSERTION",
            10: "GLOBAL_CHEAPEST_ARC",
            4: "LOCAL_CHEAPEST_INSERTION",
            5: "PATH_MOST_CONSTRAINED_ARC",
            11: "SAVINGS"
        }
        return strategy_names.get(enum_value, f"STRATEGY_{enum_value}")
    
    @staticmethod
    def _get_metaheuristic_name(enum_value):
        """Chuyển enum metaheuristic thành tên dễ đọc"""
        meta_names = {
            0: "AUTOMATIC",
            2: "GUIDED_LOCAL_SEARCH",
            3: "SIMULATED_ANNEALING",
            5: "TABU_SEARCH"
        }
        return meta_names.get(enum_value, f"METAHEURISTIC_{enum_value}")


def test_adaptive_solver():
    """Test adaptive solver với 1 dataset"""
    
    print("\n" + "="*100)
    print("TEST ADAPTIVE OR-TOOLS SOLVER")
    print("="*100 + "\n")
    
    # Test với dataset khó
    dataset_path = Path(__file__).parent.parent.parent / 'dataset' / 'R1' / 'R101.csv'
    dataset_path = str(dataset_path)
    
    solver = AdaptiveORToolsSolver(dataset_path)
    solution = solver.solve_adaptive()
    
    if solution:
        # Visualize
        solver.visualize_solution()
        
        print("\nThành công! Adaptive solver đã tự động chọn parameters phù hợp.")
    else:
        print("\nKhông tìm được solution. Có thể cần điều chỉnh rules trong DatasetAnalyzer.")


def benchmark_all_datasets():
    """Benchmark toàn bộ 56 datasets với adaptive solver"""
    
    print("\n" + "="*100)
    print("ADAPTIVE BENCHMARK - TOÀN BỘ 56 DATASETS")
    print("="*100 + "\n")
    
    dataset_base = Path(__file__).parent.parent.parent / 'dataset'
    
    # Định nghĩa các dataset types và số lượng files
    dataset_configs = {
        'C1': 9,   # C101-C109
        'C2': 8,   # C201-C208
        'R1': 12,  # R101-R112
        'R2': 11,  # R201-R211
        'RC1': 8,  # RC101-RC108
        'RC2': 8   # RC201-RC208
    }
    
    total_files = sum(dataset_configs.values())
    current_file = 0
    results = []
    
    start_time = time.time()
    
    for dataset_type, count in dataset_configs.items():
        print(f"\n{'='*100}")
        print(f"DATASET TYPE: {dataset_type}")
        print(f"{'='*100}\n")
        
        for i in range(1, count + 1):
            current_file += 1
            
            # Tạo tên file
            if dataset_type in ['C1', 'R1', 'RC1']:
                file_num = f"{i:02d}"
            else:
                file_num = f"{i:02d}"
            
            filename = f"{dataset_type[:-1]}{dataset_type[-1]}{file_num}.csv"
            dataset_path = dataset_base / dataset_type / filename
            
            if not dataset_path.exists():
                print(f"[{current_file}/{total_files}] SKIP: {filename} (file không tồn tại)")
                continue
            
            print(f"\n[{current_file}/{total_files}] Đang giải: {filename}")
            print("-" * 100)
            
            try:
                solver = AdaptiveORToolsSolver(str(dataset_path))
                
                solve_start = time.time()
                solution = solver.solve_adaptive()
                solve_time = time.time() - solve_start
                
                if solution:
                    # Lưu vào thư mục adaptive_result (khác với result)
                    result_dir = Path(__file__).parent / 'adaptive_result_1'
                    images_dir = result_dir / 'images'
                    text_dir = result_dir / 'text'
                    images_dir.mkdir(parents=True, exist_ok=True)
                    text_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Lưu visualization
                    img_path = images_dir / f"{Path(filename).stem}.png"
                    solver.visualize_solution(save_path=str(img_path))
                    
                    # Lưu file text
                    txt_path = text_dir / f"{Path(filename).stem}_adaptive.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(f"ADAPTIVE OR-TOOLS SOLVER RESULT\n")
                        f.write(f"Dataset: {filename}\n")
                        f.write(f"Status: {solution.get('status', 'N/A')}\n")
                        f.write(f"Vehicles Used: {solution['num_vehicles']}\n")
                        f.write(f"Total Distance: {solution['total_distance']:.2f}\n")
                        if solution.get('bks'):
                            f.write(f"Best Known Solution: {solution['bks']:.2f}\n")
                            if solution.get('gap') is not None:
                                f.write(f"Gap: {solution['gap']:+.2f}%\n")
                        f.write(f"Customers Served: {solution['customers_served']}\n")
                        f.write(f"Solve Time: {solve_time:.2f}s\n")
                        f.write(f"Strategy: {solver._get_strategy_name(solver.selected_strategy)}\n")
                        f.write(f"Metaheuristic: {solver._get_metaheuristic_name(solver.selected_metaheuristic)}\n\n")
                        f.write(f"Routes:\n")
                        for idx, route in enumerate(solution['routes'], 1):
                            f.write(f"Vehicle {idx}: {route}\n")
                    
                    result = {
                        'Dataset': filename,
                        'Type': dataset_type,
                        'Status': solution.get('status', 'SUCCESS'),
                        'Vehicles': solution['num_vehicles'],
                        'Total_Distance': round(solution['total_distance'], 2),
                        'BKS': solution.get('bks'),
                        'Gap_%': round(solution['gap'], 2) if solution.get('gap') is not None else None,
                        'Customers_Served': solution['customers_served'],
                        'Solve_Time': round(solve_time, 2),
                        'Strategy': solver._get_strategy_name(solver.selected_strategy),
                        'Metaheuristic': solver._get_metaheuristic_name(solver.selected_metaheuristic)
                    }
                    
                    print(f"SUCCESS - Vehicles: {result['Vehicles']}, "
                          f"Distance: {result['Total_Distance']}, "
                          f"Gap: {result['Gap_%']}%, "
                          f"Time: {result['Solve_Time']}s")
                else:
                    result = {
                        'Dataset': filename,
                        'Type': dataset_type,
                        'Status': 'FAILED',
                        'Vehicles': 0,
                        'Total_Distance': 0,
                        'BKS': BKS.get(Path(filename).stem),
                        'Gap_%': None,
                        'Customers_Served': 0,
                        'Solve_Time': round(solve_time, 2),
                        'Strategy': solver._get_strategy_name(solver.selected_strategy),
                        'Metaheuristic': solver._get_metaheuristic_name(solver.selected_metaheuristic)
                    }
                    print(f"FAILED - Không tìm được solution")
                
                results.append(result)
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                results.append({
                    'Dataset': filename,
                    'Type': dataset_type,
                    'Status': 'ERROR',
                    'Vehicles': 0,
                    'Total_Distance': 0,
                    'BKS': BKS.get(Path(filename).stem),
                    'Gap_%': None,
                    'Customers_Served': 0,
                    'Solve_Time': 0,
                    'Strategy': 'N/A',
                    'Metaheuristic': 'N/A'
                })
    
    total_time = time.time() - start_time
    
    # Tạo summary report
    print("\n" + "="*100)
    print("ADAPTIVE BENCHMARK SUMMARY")
    print("="*100 + "\n")
    
    success_count = sum(1 for r in results if r['Status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['Status'] == 'FAILED')
    error_count = sum(1 for r in results if r['Status'] == 'ERROR')
    
    print(f"Tổng số datasets: {len(results)}")
    print(f"Thành công: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Thất bại: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    print(f"Lỗi: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"Tổng thời gian: {total_time/60:.2f} phút")
    
    # Lưu kết quả vào CSV trong adaptive_result
    result_dir = Path(__file__).parent / 'adaptive_result_1'
    result_dir.mkdir(exist_ok=True)
    
    df_results = pd.DataFrame(results)
    csv_path = result_dir / 'adaptive_benchmark_summary.csv'
    df_results.to_csv(csv_path, index=False)
    
    print(f"\nĐã lưu kết quả vào: {csv_path}")

    
    # In top 5 best results
    if success_count > 0:
        print("\n" + "-"*100)
        print("TOP 5 KẾT QUẢ TỐT NHẤT (Theo Gap)")
        print("-"*100)
        df_success = df_results[df_results['Status'].str.contains('Success|Optimal|Feasible', na=False)].copy()
        
        # Sắp xếp theo Gap
        df_success['Gap_%_Safe'] = pd.to_numeric(df_success['Gap_%'], errors='coerce').fillna(999)
        top5 = df_success.nsmallest(5, 'Gap_%_Safe')
        print(top5[['Dataset', 'Vehicles', 'Total_Distance', 'BKS', 'Gap_%', 'Solve_Time']].to_string(index=False))
        
        # Thống kê về Gap
        valid_gaps = df_success['Gap_%_Safe'][df_success['Gap_%_Safe'] < 999]
        if len(valid_gaps) > 0:
            print(f"\n{'='*100}")
            print(f"THỐNG KÊ GAP")
            print(f"{'='*100}")
            print(f"Trung bình Gap: {valid_gaps.mean():.2f}%")
            print(f"Trung vị Gap: {valid_gaps.median():.2f}%")
            print(f"Gap tốt nhất: {valid_gaps.min():.2f}%")
            print(f"Gap tệ nhất: {valid_gaps.max():.2f}%")
            print(f"Solutions trong 1% BKS: {(valid_gaps <= 1).sum()}/{len(valid_gaps)} ({(valid_gaps <= 1).sum()/len(valid_gaps)*100:.1f}%)")
            print(f"Solutions trong 5% BKS: {(valid_gaps <= 5).sum()}/{len(valid_gaps)} ({(valid_gaps <= 5).sum()/len(valid_gaps)*100:.1f}%)")
            print(f"Solutions trong 10% BKS: {(valid_gaps <= 10).sum()}/{len(valid_gaps)} ({(valid_gaps <= 10).sum()/len(valid_gaps)*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    # Chạy benchmark toàn bộ 56 datasets
    benchmark_all_datasets()
    
    # Nếu muốn test 1 file, uncomment dòng dưới:
    # test_adaptive_solver()


