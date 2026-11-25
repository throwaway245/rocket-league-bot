"""Exercise the actual mechanics reward implementations with small C++ fixtures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MechanicsRewardsTest(unittest.TestCase):
    def test_event_gates_and_bounded_values(self):
        source = (ROOT / 'src/ExampleMain.cpp').read_text()
        aerial = source.split('class ProductiveAerialTouchReward : public Reward {', 1)[1].split('\n};', 1)[0]
        wavedash = source.split('class ProductiveWavedashReward : public Reward {', 1)[1].split('\n};', 1)[0]
        code = r'''
#include <algorithm>
#include <cmath>
struct Vec { float x=0; float Length() const { return std::abs(x); } };
struct Player { const Player* prev=nullptr; bool isOnGround=false, isFlipping=false, ballTouchedStep=false; Vec vel; };
struct Ball { struct { float z=0; } pos; };
struct GameState { Ball ball; float quality=0; };
struct Reward { virtual float GetReward(const Player&, const GameState&, bool) = 0; };
struct DirectionalTouchReward { float GetReward(const Player&, const GameState& s, bool) { return s.quality; } };
class ProductiveAerialTouchReward : public Reward {''' + aerial + r'''
};
class ProductiveWavedashReward : public Reward {''' + wavedash + r'''
};
int main() {
    ProductiveAerialTouchReward a; ProductiveWavedashReward w;
    Player p, prev; GameState s; s.ball.pos.z=1000; s.quality=0.8f; p.ballTouchedStep=true;
    if (std::abs(a.GetReward(p,s,false)-0.8f)>1e-5) return 1;
    p.isOnGround=true; if (a.GetReward(p,s,false)!=0) return 2;
    p.isOnGround=false; s.quality=-0.5f; if (a.GetReward(p,s,false)!=0) return 3;
    s.quality=1; s.ball.pos.z=300; if (a.GetReward(p,s,false)!=0) return 4;
    p.isOnGround=true; p.prev=&prev; prev.isFlipping=true; prev.vel.x=500; p.vel.x=930;
    if (std::abs(w.GetReward(p,s,false)-0.5f)>1e-5) return 5;
    prev.isOnGround=true; if (w.GetReward(p,s,false)!=0) return 6;
    prev.isOnGround=false; p.vel.x=550; if (w.GetReward(p,s,false)!=0) return 7;
    p.vel.x=4000; if (w.GetReward(p,s,false)!=1) return 8;
    p.prev=nullptr; if (w.GetReward(p,s,false)!=0) return 9;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.cpp'
            path.write_text(code)
            binary = Path(tmp) / 'test'
            subprocess.run(['c++', '-std=c++17', str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
