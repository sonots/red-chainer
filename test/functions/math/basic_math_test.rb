require 'chainer'
require 'cumo/narray'

class Chainer::Functions::Math::BasicMathTest < Test::Unit::TestCase
  test("Neg#forward") do
    x = Chainer::Variable.new(Cumo::DFloat[[-1, 0],[1, 2]])
    assert_equal(Cumo::DFloat[[1,0],[-1,-2]], (-x).data)
  end
end
