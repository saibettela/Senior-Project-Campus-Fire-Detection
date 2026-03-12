#include "scd41.h"

/* send a 2-byte command (MSB first) */
static HAL_StatusTypeDef scd41_send_cmd(I2C_HandleTypeDef *hi2c, uint16_t cmd)
{
    uint8_t buf[2] = { (cmd >> 8) & 0xFF, cmd & 0xFF };
    return HAL_I2C_Master_Transmit(hi2c, SCD41_I2C_ADDR, buf, 2, HAL_MAX_DELAY);
}

/* send a 2-byte command followed by one word + CRC */
static HAL_StatusTypeDef scd41_send_cmd_word(I2C_HandleTypeDef *hi2c,
                                              uint16_t cmd, uint16_t word)
{
    uint8_t buf[5];
    buf[0] = (cmd >> 8) & 0xFF;
    buf[1] =  cmd & 0xFF;
    buf[2] = (word >> 8) & 0xFF;
    buf[3] =  word & 0xFF;
    buf[4] = sensirion_crc8(&buf[2], 2);
    return HAL_I2C_Master_Transmit(hi2c, SCD41_I2C_ADDR, buf, 5, HAL_MAX_DELAY);
}

/* validate a 2-byte word + CRC triplet */
static uint8_t scd41_check_crc(const uint8_t *buf)
{
    return sensirion_crc8(buf, 2) == buf[2];
}

/* parse raw 9-byte measurement response into scd41_data_t */
static HAL_StatusTypeDef scd41_parse_measurement(const uint8_t *buf,
                                                   scd41_data_t *out)
{
    if (!scd41_check_crc(&buf[0]) ||
        !scd41_check_crc(&buf[3]) ||
        !scd41_check_crc(&buf[6])) return HAL_ERROR;

    uint16_t co2_raw = ((uint16_t)buf[0] << 8) | buf[1];
    uint16_t t_raw   = ((uint16_t)buf[3] << 8) | buf[4];
    uint16_t rh_raw  = ((uint16_t)buf[6] << 8) | buf[7];

    out->co2_ppm      = co2_raw;
    out->temperature_c = -45.0f + 175.0f * ((float)t_raw  / 65535.0f);
    out->humidity_pct  = 100.0f * ((float)rh_raw / 65535.0f);

    return HAL_OK;
}

HAL_StatusTypeDef scd41_start_periodic(I2C_HandleTypeDef *hi2c)
{
    return scd41_send_cmd(hi2c, SCD41_CMD_START_PERIODIC);
}

HAL_StatusTypeDef scd41_stop_periodic(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef ret = scd41_send_cmd(hi2c, SCD41_CMD_STOP_PERIODIC);
    if (ret != HAL_OK) return ret;
    HAL_Delay(SCD41_DELAY_STOP_MS);
    return HAL_OK;
}

HAL_StatusTypeDef scd41_data_ready(I2C_HandleTypeDef *hi2c, uint8_t *ready)
{
    uint8_t buf[3];
    HAL_StatusTypeDef ret;

    ret = scd41_send_cmd(hi2c, SCD41_CMD_DATA_READY);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SCD41_I2C_ADDR, buf, 3, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!scd41_check_crc(buf)) return HAL_ERROR;

    /* bits 10:0 == 0 means not ready */
    uint16_t word = ((uint16_t)buf[0] << 8) | buf[1];
    *ready = (word & 0x07FF) != 0;
    return HAL_OK;
}

HAL_StatusTypeDef scd41_read_measurement(I2C_HandleTypeDef *hi2c, scd41_data_t *out)
{
    uint8_t buf[9];
    HAL_StatusTypeDef ret;

    ret = scd41_send_cmd(hi2c, SCD41_CMD_READ_MEASUREMENT);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SCD41_I2C_ADDR, buf, 9, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    return scd41_parse_measurement(buf, out);
}

HAL_StatusTypeDef scd41_measure_single(I2C_HandleTypeDef *hi2c, scd41_data_t *out)
{
    uint8_t buf[9];
    HAL_StatusTypeDef ret;

    ret = scd41_send_cmd(hi2c, SCD41_CMD_MEASURE_SINGLE);
    if (ret != HAL_OK) return ret;

    HAL_Delay(SCD41_DELAY_SINGLE_SHOT_MS);

    ret = HAL_I2C_Master_Receive(hi2c, SCD41_I2C_ADDR, buf, 9, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    return scd41_parse_measurement(buf, out);
}

HAL_StatusTypeDef scd41_set_temp_offset(I2C_HandleTypeDef *hi2c, float offset_c)
{
    /* offset_ticks = offset_c * 65535 / 175 */
    uint16_t ticks = (uint16_t)(offset_c * 65535.0f / 175.0f);
    return scd41_send_cmd_word(hi2c, SCD41_CMD_SET_TEMP_OFFSET, ticks);
}

HAL_StatusTypeDef scd41_get_temp_offset(I2C_HandleTypeDef *hi2c, float *offset_c)
{
    uint8_t buf[3];
    HAL_StatusTypeDef ret;

    ret = scd41_send_cmd(hi2c, SCD41_CMD_GET_TEMP_OFFSET);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SCD41_I2C_ADDR, buf, 3, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!scd41_check_crc(buf)) return HAL_ERROR;

    uint16_t ticks = ((uint16_t)buf[0] << 8) | buf[1];
    *offset_c = 175.0f * ((float)ticks / 65535.0f);
    return HAL_OK;
}

HAL_StatusTypeDef scd41_set_pressure(I2C_HandleTypeDef *hi2c, uint32_t pressure_pa)
{
    /* datasheet: send value in Pa, valid range 70000–120000 */
    uint16_t word = (uint16_t)(pressure_pa / 100);  /* convert to hPa */
    return scd41_send_cmd_word(hi2c, SCD41_CMD_SET_PRESSURE, word);
}

HAL_StatusTypeDef scd41_forced_recal(I2C_HandleTypeDef *hi2c, uint16_t target_ppm)
{
    HAL_StatusTypeDef ret = scd41_send_cmd_word(hi2c, SCD41_CMD_FORCED_RECAL,
                                                 target_ppm);
    if (ret != HAL_OK) return ret;
    HAL_Delay(SCD41_DELAY_FORCED_RECAL_MS);
    return HAL_OK;
}

HAL_StatusTypeDef scd41_persist_settings(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef ret = scd41_send_cmd(hi2c, SCD41_CMD_PERSIST_SETTINGS);
    if (ret != HAL_OK) return ret;
    HAL_Delay(SCD41_DELAY_PERSIST_MS);
    return HAL_OK;
}

HAL_StatusTypeDef scd41_get_serial(I2C_HandleTypeDef *hi2c, uint64_t *serial)
{
    uint8_t buf[9];
    HAL_StatusTypeDef ret;

    ret = scd41_send_cmd(hi2c, SCD41_CMD_GET_SERIAL);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SCD41_I2C_ADDR, buf, 9, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!scd41_check_crc(&buf[0]) ||
        !scd41_check_crc(&buf[3]) ||
        !scd41_check_crc(&buf[6])) return HAL_ERROR;

    *serial = ((uint64_t)buf[0] << 40) | ((uint64_t)buf[1] << 32) |
              ((uint64_t)buf[3] << 24) | ((uint64_t)buf[4] << 16) |
              ((uint64_t)buf[6] <<  8) |  (uint64_t)buf[7];

    return HAL_OK;
}

HAL_StatusTypeDef scd41_factory_reset(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef ret = scd41_send_cmd(hi2c, SCD41_CMD_FACTORY_RESET);
    if (ret != HAL_OK) return ret;
    HAL_Delay(SCD41_DELAY_FACTORY_RESET_MS);
    return HAL_OK;
}

HAL_StatusTypeDef scd41_reinit(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef ret = scd41_send_cmd(hi2c, SCD41_CMD_REINIT);
    if (ret != HAL_OK) return ret;
    HAL_Delay(SCD41_DELAY_REINIT_MS);
    return HAL_OK;
}
