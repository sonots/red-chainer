# frozen_string_literal: true

class Chainer::Utils::ConvTest < Test::Unit::TestCase
  data({
    test1: {
      case: { size: 4, k: 2, s: 1, p: 1, options: {} },
      expected: 5
    },
    test2: {
      case: { size: 4, k: 2, s: 2, p: 1, options: {} },
      expected: 3
    },
    test3: {
      case: { size: 4, k: 2, s: 2, p: 2, options: {} },
      expected: 4
    },
    test4: {
      case: { size: 4, k: 2, s: 2, p: 2, options: { cover_all: true } },
      expected: 4
    },
    test5: {
      case: { size: 4, k: 2, s: 2, p: 2, options: { cover_all: true, d: 3 } },
      expected: 3
    },
  })
  def test_get_conv_outsize(data)
    test_case = data[:case]
    actual = Chainer::Utils::Conv.get_conv_outsize(test_case[:size], test_case[:k], test_case[:s], test_case[:p], **test_case[:options])
    assert_equal(data[:expected], actual)
  end

  data({
    test1: {
      case: {
        img: Cumo::DFloat.new(1, 2, 4, 4).seq, kh: 2, kw: 2, sy: 1, sx: 1, ph: 0, pw: 0, options: {}
      },
      expected: Cumo::DFloat[[[[[[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [8.0, 9.0, 10.0]], [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]], [[[4.0, 5.0, 6.0], [8.0, 9.0, 10.0], [12.0, 13.0, 14.0]], [[5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [13.0, 14.0, 15.0]]]], [[[[16.0, 17.0, 18.0], [20.0, 21.0, 22.0], [24.0, 25.0, 26.0]], [[17.0, 18.0, 19.0], [21.0, 22.0, 23.0], [25.0, 26.0, 27.0]]], [[[20.0, 21.0, 22.0], [24.0, 25.0, 26.0], [28.0, 29.0, 30.0]], [[21.0, 22.0, 23.0], [25.0, 26.0, 27.0], [29.0, 30.0, 31.0]]]]]] 
    },
    test2: {
      case: {
        img: Cumo::DFloat.new(2, 2, 4, 4).seq, kh: 2, kw: 2, sy: 1, sx: 1, ph: 0, pw: 0, options: {}
      },
      expected: Cumo::DFloat[[[[[[0.0, 1.0, 2.0], [4.0, 5.0, 6.0], [8.0, 9.0, 10.0]], [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]], [[[4.0, 5.0, 6.0], [8.0, 9.0, 10.0], [12.0, 13.0, 14.0]], [[5.0, 6.0, 7.0], [9.0, 10.0, 11.0], [13.0, 14.0, 15.0]]]], [[[[16.0, 17.0, 18.0], [20.0, 21.0, 22.0], [24.0, 25.0, 26.0]], [[17.0, 18.0, 19.0], [21.0, 22.0, 23.0], [25.0, 26.0, 27.0]]], [[[20.0, 21.0, 22.0], [24.0, 25.0, 26.0], [28.0, 29.0, 30.0]], [[21.0, 22.0, 23.0], [25.0, 26.0, 27.0], [29.0, 30.0, 31.0]]]]], [[[[[32.0, 33.0, 34.0], [36.0, 37.0, 38.0], [40.0, 41.0, 42.0]], [[33.0, 34.0, 35.0], [37.0, 38.0, 39.0], [41.0, 42.0, 43.0]]], [[[36.0, 37.0, 38.0], [40.0, 41.0, 42.0], [44.0, 45.0, 46.0]], [[37.0, 38.0, 39.0], [41.0, 42.0, 43.0], [45.0, 46.0, 47.0]]]], [[[[48.0, 49.0, 50.0], [52.0, 53.0, 54.0], [56.0, 57.0, 58.0]], [[49.0, 50.0, 51.0], [53.0, 54.0, 55.0], [57.0, 58.0, 59.0]]], [[[52.0, 53.0, 54.0], [56.0, 57.0, 58.0], [60.0, 61.0, 62.0]], [[53.0, 54.0, 55.0], [57.0, 58.0, 59.0], [61.0, 62.0, 63.0]]]]]]
    },
    test3: {
      case: {
        img: Cumo::DFloat.new(1, 1, 4, 4).seq, kh: 2, kw: 2, sy: 2, sx: 2, ph: 0, pw: 0, options: {}
      },
      expected: Cumo::DFloat[[[[[[0.0, 2.0], [8.0, 10.0]], [[1.0, 3.0], [9.0, 11.0]]], [[[4.0, 6.0], [12.0, 14.0]], [[5.0, 7.0], [13.0, 15.0]]]]]]
    },
    test4: {
      case: {
        img: Cumo::DFloat.new(1, 1, 4, 4).seq, kh: 2, kw: 2, sy: 2, sx: 2, ph: 1, pw: 1, options: {}
      },
      expected: Cumo::DFloat[[[[[[0.0, 0.0, 0.0], [0.0, 5.0, 7.0], [0.0, 13.0, 15.0]], [[0.0, 0.0, 0.0], [4.0, 6.0, 0.0], [12.0, 14.0, 0.0]]], [[[0.0, 1.0, 3.0], [0.0, 9.0, 11.0], [0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [8.0, 10.0, 0.0], [0.0, 0.0, 0.0]]]]]]
    },
    test5: {
      case: {
        img: Cumo::DFloat.new(1, 1, 4, 4).seq, kh: 2, kw: 2, sy: 2, sx: 2, ph: 1, pw: 1, options: { pval: 3 }
      },
      expected: Cumo::DFloat[[[[[[3.0, 3.0, 3.0], [3.0, 5.0, 7.0], [3.0, 13.0, 15.0]], [[3.0, 3.0, 3.0], [4.0, 6.0, 3.0], [12.0, 14.0, 3.0]]], [[[3.0, 1.0, 3.0], [3.0, 9.0, 11.0], [3.0, 3.0, 3.0]], [[0.0, 2.0, 3.0], [8.0, 10.0, 3.0], [3.0, 3.0, 3.0]]]]]]
    },
    test6: {
      case: {
        img: Cumo::DFloat.new(1, 1, 4, 4).seq, kh: 2, kw: 2, sy: 2, sx: 2, ph: 0, pw: 0, options: {}
      },
      expected: Cumo::DFloat[[[[[[0.0, 2.0], [8.0, 10.0]], [[1.0, 3.0], [9.0, 11.0]]], [[[4.0, 6.0], [12.0, 14.0]], [[5.0, 7.0], [13.0, 15.0]]]]]]
    },
    test7: { # cover_all
      case: {
        img: Cumo::DFloat.new(1, 1, 4, 6).seq, kh: 2, kw: 2, sy: 3, sx: 3, ph: 0, pw: 0, options: { cover_all: true }
      },
      expected: Cumo::DFloat[[[[[[0.0, 3.0, 0.0], [18.0, 21.0, 0.0]], [[1.0, 4.0, 0.0], [19.0, 22.0, 0.0]]], [[[6.0, 9.0, 0.0], [0.0, 0.0, 0.0]], [[7.0, 10.0, 0.0], [0.0, 0.0, 0.0]]]]]]
    }
  })
  def test_im2col_cpu(data)
    test_case = data[:case]
    actual = Chainer::Utils::Conv.im2col_cpu(test_case[:img], test_case[:kh], test_case[:kw], test_case[:sy], test_case[:sx], test_case[:ph], test_case[:pw], **test_case[:options])
    assert_equal(data[:expected], actual)
  end

  data({
    test1: {
      case: {
        col: Cumo::DFloat[[[[[[0.0, 2.0], [8.0, 10.0]], [[1.0, 3.0], [9.0, 11.0]]], [[[4.0, 6.0], [12.0, 14.0]], [[5.0, 7.0], [13.0, 15.0]]]]]],
        sy: 2, sx: 2, ph: 1, pw: 1, h: 2, w:2 
      },
      expected: Cumo::DFloat[[[[5.0, 6.0], [9.0, 10.0]]]]
    },
    test2: {
      case: {
        col: Cumo::DFloat[[[[[[0.0, 3.0, 0.0], [18.0, 21.0, 0.0]], [[1.0, 4.0, 0.0], [19.0, 22.0, 0.0]]], [[[6.0, 9.0, 0.0], [0.0, 0.0, 0.0]], [[7.0, 10.0, 0.0], [0.0, 0.0, 0.0]]]]]],
        sy: 1, sx: 1, ph: 1, pw: 1, h: 2, w:2
      },
      expected: Cumo::DFloat[[[[56.0, 32.0], [0.0, 0.0]]]]
    },
    test3: {
      case: {
        col: Cumo::DFloat.new(2, 2, 2, 2, 3, 4).seq,
        sy: 2, sx: 2, ph: 2, pw: 2, h: 3, w: 4
      },
      expected: Cumo::DFloat[[[[5.0, 17.0, 6.0, 18.0], [29.0, 41.0, 30.0, 42.0], [9.0, 21.0, 10.0, 22.0]], [[53.0, 65.0, 54.0, 66.0], [77.0, 89.0, 78.0, 90.0], [57.0, 69.0, 58.0, 70.0]]], [[[101.0, 113.0, 102.0, 114.0], [125.0, 137.0, 126.0, 138.0], [105.0, 117.0, 106.0, 118.0]], [[149.0, 161.0, 150.0, 162.0], [173.0, 185.0, 174.0, 186.0], [153.0, 165.0, 154.0, 166.0]]]],
    }
  })
  def test_col2im_cpu(data)
    test_case = data[:case]
    actual = Chainer::Utils::Conv.col2im_cpu(test_case[:col], test_case[:sy], test_case[:sx], test_case[:ph], test_case[:pw], test_case[:h], test_case[:w])
    assert_equal(data[:expected], actual)
  end
end
