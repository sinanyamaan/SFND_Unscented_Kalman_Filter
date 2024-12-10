#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

    is_initialized_ = false;

    time_us_ = 0;

    n_x_ = 5;

    n_aug_ = 7;

    Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

    weights_ = VectorXd::Zero(Xsig_pred_.cols());

    lambda_ = 3.0 - n_aug_;

    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {
        weights_(i) = 0.5 / (n_aug_ + lambda_);
    }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
    if (!is_initialized_)
    {
        if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
        {
            x_(0) = meas_package.raw_measurements_(0);
            x_(1) = meas_package.raw_measurements_(1);
            P_(0, 0) = std_laspx_ * std_laspx_;
            P_(1, 1) = std_laspy_ * std_laspy_;

            is_initialized_ = true;
        }
        else if ( use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            const auto range = meas_package.raw_measurements_(0);
            const auto azimuth = meas_package.raw_measurements_(1);
            const auto radial_velocity = meas_package.raw_measurements_(2);

            const auto px = range * cos(azimuth);
            const auto py = range * sin(azimuth);
            const auto vx = radial_velocity * cos(azimuth);
            const auto vy = radial_velocity * sin(azimuth);

            x_(0) = px;
            x_(1) = py;
            x_(2) = sqrt(vx * vx + vy * vy);

            is_initialized_ = true;
        }
        else
        {
            return;
        }

        time_us_ = meas_package.timestamp_;
        return;
    }

    const auto dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        UpdateLidar(meas_package);
    }
    else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        UpdateRadar(meas_package);
    }
}

void UKF::Prediction(const double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

    Eigen::VectorXd x_aug = Eigen::VectorXd::Zero(n_aug_);
    Eigen::MatrixXd P_aug  = Eigen::MatrixXd::Zero(n_aug_, n_aug_);

    Eigen::MatrixXd Xsig_aug_ = Eigen::MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

    x_aug.head(n_x_) = x_;

    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    Eigen::MatrixXd L_aug = P_aug.llt().matrixL();

    // Create sigma points in augmented state space
    Xsig_aug_.col(0) = x_aug;
    for (auto i = 0; i < n_aug_; i++)
    {
        Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L_aug.col(i);
        Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L_aug.col(i);
    }

    // Predict motion of sigma points
    Xsig_pred_.fill(0.0);

    for (auto i = 0; i < Xsig_aug_.cols(); i++)
    {
        const double px = Xsig_aug_(0, i);
        const double py = Xsig_aug_(1, i);
        const double v = Xsig_aug_(2, i);
        const double yaw = Xsig_aug_(3, i);
        const double yawd = Xsig_aug_(4, i);
        const double nu_a = Xsig_aug_(5, i);
        const double nu_yawdd = Xsig_aug_(6, i);

        const double cos_yaw = cos(yaw);
        const double sin_yaw = sin(yaw);

        if (std::abs(yawd) <= 1e-3 )
        {
            Xsig_pred_(0, i) = px + v * cos_yaw * delta_t + delta_t * delta_t / 2 * cos_yaw * nu_a;
            Xsig_pred_(1, i) = py + v * sin(yaw) * delta_t + delta_t * delta_t / 2 * sin_yaw * nu_a;
        }
        else
        {
            Xsig_pred_(0, i) = px + v / yawd * (sin(yaw + yawd * delta_t) - sin_yaw) +
                delta_t * delta_t / 2 * cos_yaw * nu_a;
            Xsig_pred_(1, i) = py + v / yawd * (-cos(yaw + yawd * delta_t) + cos_yaw) +
                              delta_t * delta_t / 2 * sin_yaw * nu_a;
        }

        Xsig_pred_(2, i) = v + delta_t * nu_a;
        Xsig_pred_(3, i) = yaw + yawd * delta_t + delta_t * delta_t / 2 * nu_yawdd;
        Xsig_pred_(4, i) = yawd + delta_t * nu_yawdd;
    }

    VectorXd x_pred = VectorXd::Zero(n_x_);
    MatrixXd P_pred = MatrixXd::Zero(n_x_, n_x_);

    for (int i = 0; i < Xsig_pred_.cols(); i++) {
        x_pred = x_pred + weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < Xsig_pred_.cols(); i++) {
        P_pred = P_pred + weights_(i) *
                (Xsig_pred_.col(i) - x_) *
                (Xsig_pred_.col(i) - x_).transpose();
    }

    x_ = x_pred;
    P_ = P_pred;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    const int n_z = meas_package.raw_measurements_.size();

    const auto px = meas_package.raw_measurements_(0);
    const auto py = meas_package.raw_measurements_(1);

    auto z = Eigen::VectorXd(n_z);
    z << px, py;

    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n_z, n_z);
    R(0, 0) = std_laspx_ * std_laspx_;
    R(1, 1) = std_laspy_ * std_laspy_;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_z, n_x_);
    H(0, 0) = 1;
    H(1, 1) = 1;

    const Eigen::VectorXd z_pred = H * x_;
    Eigen::VectorXd y = z - z_pred;

    Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

    x_ = x_ + K * y;
    P_ = (MatrixXd::Identity(n_x_, n_x_) - K * H) * P_;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

    const int n_z = meas_package.raw_measurements_.size();

    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        const double p_x = Xsig_pred_(0, i);
        const double p_y = Xsig_pred_(1, i);
        const double v = Xsig_pred_(2, i);
        const double yaw = Xsig_pred_(3, i);

        const double v1 = cos(yaw) * v;
        const double v2 = sin(yaw) * v;

        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);
    }

    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // Angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd::Zero(n_z, n_z);
    R(0, 0) = std_radr_ * std_radr_;
    R(1, 1) = std_radphi_ * std_radphi_;
    R(2, 2) = std_radrd_ * std_radrd_;
    S += R;

    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();

    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_diff = z - z_pred;
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
}