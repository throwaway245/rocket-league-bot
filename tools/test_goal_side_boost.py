"""Exercise the actual C++ goal-side boost reward for both teams."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class GoalSideBoostTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_goal_side_recovery(self):
        source = (pathlib.Path(__file__).resolve().parents[1] / 'src/ExampleMain.cpp').read_text()
        reward = source.split('class GoalSideBoostReward : public Reward {', 1)[1].split('\n};', 1)[0]
        program = r'''
#include <algorithm>
#include <cmath>
#define RS_MIN(a,b) std::min((a),(b))
#define RS_CLAMP(x,a,b) std::clamp((x),(a),(b))
enum class Team { BLUE, ORANGE };
struct Vec { float y; };
struct Player { Team team; Vec pos, vel; float boost; };
struct GameState { Vec ballPos; struct { Vec pos; } ball; };
struct Reward { virtual float GetReward(const Player&, const GameState&, bool) = 0; };
namespace CommonValues { constexpr float CAR_MAX_SPEED = 2300.f; }
class GoalSideBoostReward : public Reward {
''' + reward + r'''
};
int main() {
    GoalSideBoostReward reward;
    for (auto team : {Team::BLUE, Team::ORANGE}) {
        float sign = team == Team::BLUE ? 1.f : -1.f;
        GameState state{}; state.ball.pos.y = -2000.f * sign;
        Player p{team, {-4000.f * sign}, {-1200.f * sign}, 100.f};
        if (std::abs(reward.GetReward(p, state, false) - 1200.f / 2300.f) > 0.0001f) return 1;
        p.boost = 25.f;
        if (std::abs(reward.GetReward(p, state, false) - 600.f / 2300.f) > 0.0001f) return 2;
        p.pos.y = 0.f;
        if (reward.GetReward(p, state, false) != 0.f) return 3;
        p.pos.y = -4000.f * sign; p.vel.y = 1200.f * sign;
        if (reward.GetReward(p, state, false) != 0.f) return 4;
        p.vel.y = 0.f;
        if (reward.GetReward(p, state, false) != 0.f) return 5;
    }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)
            (path / 'test.cpp').write_text(program)
            subprocess.run(['c++', '-std=c++17', str(path / 'test.cpp'), '-o', str(path / 'test')], check=True)
            subprocess.run([str(path / 'test')], check=True)


if __name__ == '__main__':
    unittest.main()
