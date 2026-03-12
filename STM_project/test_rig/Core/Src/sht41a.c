#include "sht41a.h"

/* send a single command byte */
static HAL_StatusTypeDef sht41a_send_cmd(I2C_HandleTypeDef *hi2c, uint8_t cmd)
{
    return HAL_I2C_Master_Transmit(hi2c, SHT41A_I2C_ADDR, &cmd, 1, HAL_MAX_DELAY);
}

/* validate a 2-byte word + CRC triplet */
static uint8_t sht41a_check_crc(const uint8_t *buf)
{
    return sensirion_crc8(buf, 2) == buf[2];
}

HAL_StatusTypeDef sht41a_measure(I2C_HandleTypeDef *hi2c, uint8_t cmd,
                                  uint32_t delay_ms, sht41a_data_t *out)
{
    uint8_t buf[6];
    HAL_StatusTypeDef ret;

    ret = sht41a_send_cmd(hi2c, cmd);
    if (ret != HAL_OK) return ret;

    HAL_Delay(delay_ms);

    ret = HAL_I2C_Master_Receive(hi2c, SHT41A_I2C_ADDR, buf, 6, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sht41a_check_crc(&buf[0]) || !sht41a_check_crc(&buf[3])) return HAL_ERROR;

    uint16_t t_raw = (buf[0] << 8) | buf[1];
    uint16_t rh_raw = (buf[3] << 8) | buf[4];

    out->temperature_c = -45.0f + 175.0f * ((float)t_raw / 65535.0f);
    out->humidity_pct  = -6.0f  + 125.0f * ((float)rh_raw / 65535.0f);

    /* clamp humidity to [0, 100] */
    if (out->humidity_pct < 0.0f)   out->humidity_pct = 0.0f;
    if (out->humidity_pct > 100.0f) out->humidity_pct = 100.0f;

    return HAL_OK;
}

HAL_StatusTypeDef sht41a_get_serial(I2C_HandleTypeDef *hi2c, uint32_t *serial)
{
    uint8_t buf[6];
    HAL_StatusTypeDef ret;

    ret = sht41a_send_cmd(hi2c, SHT41A_CMD_SERIAL);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SHT41A_I2C_ADDR, buf, 6, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sht41a_check_crc(&buf[0]) || !sht41a_check_crc(&buf[3])) return HAL_ERROR;

    *serial = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
              ((uint32_t)buf[3] << 8)  |  (uint32_t)buf[4];

    return HAL_OK;
}

HAL_StatusTypeDef sht41a_reset(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef ret = sht41a_send_cmd(hi2c, SHT41A_CMD_RESET);
    if (ret != HAL_OK) return ret;
    HAL_Delay(1);
    return HAL_OK;
}
