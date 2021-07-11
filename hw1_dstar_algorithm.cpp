#include <iostream>
#include <cstdio>
#include <vector>
#include <queue>
#include <list>
#include <windows.h>
#include <cstdlib>
#include <ctime>

namespace DStar {
    
    struct Coord {
        int x;
        int y;
        const bool operator==(const Coord& p) const {
            return p.x == x && p.y == y;
        }
        const bool operator!=(const Coord& p) const {
            return p.x != x || p.y != y;
        }
    };
    
    struct CoordExt {
        int x;
        int y;
        int val;
        const bool operator==(const CoordExt& p) const {
            return p.x == x && p.y == y;
        }
        const bool operator!=(const CoordExt& p) const {
            return p.x != x || p.y != y;
        }
        const bool operator <(const CoordExt& p)const { //Used to Make Heap
            return val > p.val;
        }
    };
    
    template<class T>class PriorityQueue :public std::priority_queue<T, std::vector<T>> {
        public:
        bool remove(T value) {
            auto it = std::find(this->c.begin(), this->c.end(), value);
            if (it != this->c.end()) {
                this->c.erase(it);
                std::make_heap(this->c.begin(), this->c.end(),this->comp);
                return true;
            }
            else {
                std::cout << "REMOVE FAILED" << std::endl;
                return false;
            }
        }
    };
    
    class SquareLabyrinth {
        public:
        int** matrix;
        int** k;
        int** h;
        int** t;
        Coord** b;
        
        int rows;
        int cols;
        int route_diag = 0;
        int route_adjacent = 0;
        int route_blocked = 0;
        
        const int S_NEW = 1;
        const int S_OPEN = 2;
        const int S_CLOSED = 3;
        public:
        SquareLabyrinth(int _rows, int _cols,int _rdiag,int _radj,int _rblk) {
            rows = _rows;
            cols = _cols;
            route_diag = _rdiag;
            route_adjacent = _radj;
            route_blocked = _rblk;
            matrix = new int* [_rows];
            k = new int* [_rows];
            t = new int* [_rows];
            h = new int* [_rows];
            b = new Coord* [_rows];
            for (int i = 0; i < rows; i++) {
                matrix[i] = new int[_cols];
                k[i] = new int[_cols];
                h[i] = new int[_cols];
                t[i] = new int[_cols];
                b[i] = new Coord[_cols];
                for (int j = 0; j < cols; j++)
                {
                    matrix[i][j] = 0;
                    k[i][j] = 999999999;
                    h[i][j] = 999999999;
                    t[i][j] = S_NEW;
                    b[i][j] = { -1,-1 };
                }
            }
        }
        ~SquareLabyrinth() {
            for (int i = 0; i < rows; i++) {
                delete[] matrix[i];
                delete[] h[i];
                delete[] t[i];
                delete[] k[i];
                delete[] b[i];
            }
            delete[] matrix;
            delete[] k;
            delete[] b;
            delete[] t;
            delete[] h;
        }
        int SetBlock(int r, int c,int type) {
            matrix[r][c] = type;
            return 1;
        }
        int GetBlock(int r, int c) {
            return matrix[r][c];
        }
        int GetT(Coord v) {
            return t[v.x][v.y];
        }
        int GetH(Coord v) {
            return h[v.x][v.y];
        }
        int GetK(Coord v) {
            return k[v.x][v.y];
        }
        Coord GetB(Coord v) {
            return b[v.x][v.y];
        }
        void SetB(Coord dst, Coord val) {
            b[dst.x][dst.y] = val;
        }
        void SetH(Coord dst, int val) {
            h[dst.x][dst.y] = val;
        }
        void OutputTrack(Coord S) {
            while (S.x != -1) {
                std::cout << S.x << "," << S.y << std::endl;
                S = GetB(S);
            }
        }
        int GetDist(Coord x,Coord y) {
            if (matrix[x.x][x.y] == 1 || matrix[y.x][y.y] == 1) return route_blocked;
            if (matrix[x.x][x.y] == 4 || matrix[y.x][y.y] == 4) return route_blocked;
            if (x.x == y.x || x.y == y.y) return route_adjacent;
            return route_diag;
        }
        void OutputMatrix(Coord start,Coord cur) {
            std::cout << "D* Path Planning - 1950641" << std::endl;
            HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
            Coord S = start;
            int cnt = 0;
            while (S.x != -1) {
                if (GetBlock(S.x, S.y) != 2) {
                    SetBlock(S.x, S.y, -1);
                }
                cnt++;
                if (cnt > 99999) {
                    std::cout << "No Available Plans                        " << std::endl;
                    break;
                }
                S = GetB(S);
                
            }
            //Replace characters to filled squares (instead of sharps) here if your computer support the charset
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (i == cur.x && j == cur.y) { //Current Place
                        SetConsoleTextAttribute(handle, FOREGROUND_RED | FOREGROUND_GREEN );
                        std::cout << "#";
                    }
                    else if (matrix[i][j] == -1) { //Planned Routes
                        SetConsoleTextAttribute(handle, FOREGROUND_INTENSITY | FOREGROUND_GREEN);
                        std::cout << "#";
                    }
                    else if (matrix[i][j] == 0) { //Accessible Area
                        SetConsoleTextAttribute(handle, FOREGROUND_INTENSITY | FOREGROUND_BLUE);
                        std::cout << "#";
                    }
                    else if (matrix[i][j] == 1) {//Blocked Area
                        SetConsoleTextAttribute(handle, FOREGROUND_INTENSITY | FOREGROUND_RED);
                        std::cout << "#";
                    }
                    else if (matrix[i][j] == 2) {//Visited Area
                        SetConsoleTextAttribute(handle, FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
                        std::cout << "#";
                    }
                    else if (matrix[i][j] == 4) {//Blocked Area
                        SetConsoleTextAttribute(handle,  FOREGROUND_RED );
                        std::cout << "#";
                    }
                }
                std::cout << std::endl;
            }
            SetConsoleTextAttribute(handle, FOREGROUND_INTENSITY | FOREGROUND_BLUE);
            S = start;
            while (S.x != -1) {
                if (GetBlock(S.x, S.y) != 2) {
                    SetBlock(S.x, S.y, 0);
                }
                S = GetB(S);
            }
        }
    };
    
    
    
    class DynamicStar {
        public:
        SquareLabyrinth* graph = nullptr;
        PriorityQueue<CoordExt> openList;
        std::list<Coord> openList2;
        
        Coord MinState(int& k_min) {
            int ansV = 999999999;
            Coord ans = { 0,0 };
            Coord x = { 0,0 };
            if (!openList.empty()) {
                CoordExt tmp = openList.top();
                ans = { tmp.x,tmp.y };
                ansV = tmp.val;
            }
            k_min = ansV;
            return ans;
        }
        
        int Delete(Coord t) {
            //openList.remove({ t.x,t.y,graph->GetK(t) });
            openList.pop();
            graph->t[t.x][t.y] = graph->S_CLOSED;
            return 1;
        }
        
        int Min(int x, int y) {
            return (x > y) ? y : x;
        }
        
        int Insert(Coord t, int hnew) {
            switch (graph->t[t.x][t.y]) {
                case 1: //New
                graph->k[t.x][t.y] = Min(hnew, graph->k[t.x][t.y]);
                graph->t[t.x][t.y] = graph->S_OPEN;
                openList.push({ t.x,t.y,graph->k[t.x][t.y] });
                break;
                case 2://Open
                //std::cout << "BEFORE REMOVE" << openList.size() << std::endl;
                openList.remove({ t.x,t.y,graph->k[t.x][t.y] });
                //std::cout << "AFTER REMOVE" << openList.size() << std::endl;
                graph->k[t.x][t.y] = Min(graph->h[t.x][t.y], hnew);
                openList.push({ t.x,t.y,graph->k[t.x][t.y] });
                break;
                case 3://Closed
                graph->h[t.x][t.y] = hnew;
                graph->t[t.x][t.y] = graph->S_OPEN;
                openList.push({ t.x,t.y,graph->k[t.x][t.y] });
                break;
            }
            return 1;
        }
        
        
        //Main
        int ProcS_0(Coord X, Coord Y, int K_old) {
            if (graph->GetH(Y) < K_old && graph->GetH(X) > graph->GetH(Y) + graph->GetDist(X, Y)) {
                graph->SetB(X, Y);
                graph->SetH(X, graph->GetH(Y) + graph->GetDist(X, Y));
            }
            return 1;
        }
        int ProcS_1(Coord X,Coord Y) {
            if (graph->GetT(Y) == graph->S_NEW ||
            (graph->GetB(Y) == X && graph->GetH(Y) != graph->GetH(X) + graph->GetDist(X, Y)) ||
            (graph->GetB(Y) != X && graph->GetH(Y) > graph->GetH(X) + graph->GetDist(X, Y))) {
                graph->SetB(Y, X);
                Insert(Y, graph->GetH(X) + graph->GetDist(X, Y));
            }
            
            return 1;
        }
        int ProcS_2(Coord X, Coord Y, int K_old) {
            //std::cout << "IN PROCS2" << std::endl;
            if (graph->GetT(Y) == graph->S_NEW ||
            (graph->GetB(Y) == X && graph->GetH(Y) != graph->GetH(X) + graph->GetDist(X, Y)) ||
            (graph->GetB(Y) != X && graph->GetH(Y) > graph->GetH(X) + graph->GetDist(X, Y))) {
                graph->SetB(Y, X);
                Insert(Y, graph->GetH(X) + graph->GetDist(X, Y));
            }
            else {
                if (graph->GetB(Y) != X && graph->GetH(Y) > graph->GetH(X) + graph->GetDist(X, Y)) {
                    Insert(X, graph->GetH(X));
                }
                else {
                    if (graph->GetB(Y) != X &&
                    graph->GetH(X) > graph->GetH(Y) + graph->GetDist(X, Y) &&
                    graph->GetT(Y) == graph->S_CLOSED &&
                    graph->GetH(Y) > K_old) {
                        
                        Insert(Y, graph->GetH(Y));
                    }
                }
            }
            return 1;
        }
        std::vector<Coord> GenerateAdjacentSuccessor(Coord X) {
            std::vector<Coord> ans;
            if (X.x - 1 >= 0)ans.push_back({ X.x - 1,X.y });
            if (X.x + 1 < graph->rows)ans.push_back({ X.x + 1,X.y });
            if (X.y + 1 < graph->cols)ans.push_back({ X.x,X.y + 1 });
            if (X.y - 1 >= 0)ans.push_back({ X.x,X.y - 1 });
            if (X.x - 1 >= 0 && X.y - 1 >= 0) ans.push_back({ X.x - 1,X.y - 1 });
            if (X.x - 1 >= 0 && X.y + 1 < graph->cols) ans.push_back({ X.x - 1,X.y + 1 });
            if (X.x + 1 < graph->rows && X.y + 1 < graph->cols) ans.push_back({ X.x + 1,X.y + 1 });
            if (X.x + 1 < graph->rows && X.y - 1 >= 0) ans.push_back({ X.x + 1,X.y - 1 });
            return ans;
        }
        int Process_State() {
            int K_old;
            Coord X = MinState(K_old);
            if (K_old == 999999999) {
                return -1;
            }
            Delete(X);
            if (K_old < graph->h[X.x][X.y]) {
                std::vector<Coord> YSet = GenerateAdjacentSuccessor(X);
                for (int i = 0; i < YSet.size(); i++) {
                    ProcS_0(X, YSet[i], K_old);
                }
                
            }
            if (K_old == graph->h[X.x][X.y]) {
                std::vector<Coord> YSet = GenerateAdjacentSuccessor(X);
                for (int i = 0; i < YSet.size(); i++) {
                    ProcS_1(X, YSet[i]);
                }
            }
            else {
                std::vector<Coord> YSet = GenerateAdjacentSuccessor(X);
                for (int i = 0; i < YSet.size(); i++) {
                    ProcS_2(X, YSet[i], K_old);
                }
            }
            MinState(K_old);
            return K_old;
        }
        
        int ModifyCost(Coord X,int Type) {
            graph->SetBlock(X.x, X.y, Type);
            if (graph->GetT(X) == graph->S_CLOSED) {
                Insert(X, graph->GetH(graph->GetB(X)) + graph->GetDist(X, graph->GetB(X)));
            }
            if (X.x - 1 >= 0) {
                Coord Y = { X.x - 1,X.y };
                if (graph->GetT(X) == graph->S_CLOSED) {
                    Insert(Y, graph->GetH(graph->GetB(Y)) + graph->GetDist(Y, graph->GetB(Y)));
                }
            }
            //Right
            if (X.x + 1 < graph->rows) {
                Coord Y = { X.x + 1,X.y };
                if (graph->GetT(X) == graph->S_CLOSED) {
                    Insert(Y, graph->GetH(graph->GetB(Y)) + graph->GetDist(Y, graph->GetB(Y)));
                }
            }
            //Down
            if (X.y + 1 < graph->cols) {
                Coord Y = { X.x ,X.y + 1 };
                if (graph->GetT(X) == graph->S_CLOSED) {
                    Insert(Y, graph->GetH(graph->GetB(Y)) + graph->GetDist(Y, graph->GetB(Y)));
                }
            }
            //Up
            if (X.y - 1 >= 0) {
                Coord Y = { X.x,X.y - 1 };
                if (graph->GetT(X) == graph->S_CLOSED) {
                    Insert(Y, graph->GetH(graph->GetB(Y)) + graph->GetDist(Y, graph->GetB(Y)));
                }
            }
            return 1;
        }
        
        int Start(Coord G) {
            graph->SetH(G, 0);
            Insert(G, 0);
            while (Process_State() != -1);
            return 1;
        }
        
        int StartAg() {
            while (Process_State() != -1);
            return 1;
        }
    };
    
    class DStarDemonstrator {
        public:
        bool cls(){
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            COORD coordScreen = { 0, 0 };   
            DWORD cCharsWritten;
            CONSOLE_SCREEN_BUFFER_INFO csbi; 
            DWORD dwConSize;                 
            if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
            return false;
            dwConSize = csbi.dwSize.X * csbi.dwSize.Y;
            if (!GetConsoleScreenBufferInfo(hConsole, &csbi))
            return false;
            if (!FillConsoleOutputAttribute(hConsole, csbi.wAttributes, dwConSize, coordScreen, &cCharsWritten))
            return false;
            if (!SetConsoleCursorPosition(hConsole, coordScreen))
            return false;
        }
        void show(Coord sz,int obstacle_attempts) {
            DStar::DynamicStar w;
            w.graph = new DStar::SquareLabyrinth(sz.x, sz.y,1414, 1000, 99999);
            for (int i = 0; i <= obstacle_attempts; i++) {
                int rx = rand() % w.graph->rows;
                int ry = rand() % w.graph->cols;
                if (rx == w.graph->rows - 1 && ry == w.graph->cols - 1)continue;
                if (rx == 0 && ry == 0)continue;
                w.graph->SetBlock(rx, ry, 4);
            }
            w.Start({ sz.x - 1,sz.y - 1 });
            w.graph->OutputTrack({ 0,0 });
            Coord S = { 0,0 }, N = { 0,0 };
            system("cls");
            int setBarrierFlag = 0;
            while (S.x != -1) {
                setBarrierFlag = (rand() % 2 != 0);
                if (S.y >= w.graph->cols - 4 && S.x >= w.graph->rows - 4) {
                    setBarrierFlag = 0;
                }
                cls();
                if (S.y <4 && S.x <4) {
                    setBarrierFlag = 0;
                }
                if (setBarrierFlag) {
                    Coord Nxt = w.graph->GetB(S);
                    if (Nxt.x != -1) {
                        w.ModifyCost(Nxt, 1);
                        w.StartAg();
                        N = S;
                    }
                    else {
                        setBarrierFlag = 0;
                    }
                }
                w.graph->SetBlock(S.x, S.y, 2);
                w.graph->OutputMatrix(N,S);
                std::cout << "Now At:" << S.x << "," << S.y <<"                 "<< std::endl;
                if (setBarrierFlag) {
                    std::cout << "A New Obstacle Emerges" << std::endl;
                }
                else {
                    std::cout << "                             " << std::endl;
                }
                Sleep(1000);
                S = w.graph->GetB(S);
            }
            
        }
    };
}



int main() {
    srand(time(NULL));
    DStar::DStarDemonstrator p;
    p.show({ 20,20 }, 30);
}
