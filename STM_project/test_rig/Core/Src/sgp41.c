#include "sgp41.h"

/* write a 2-byte command (MSB first) */
static HAL_StatusTypeDef sgp41_send_cmd(I2C_HandleTypeDef *hi2c, uint16_t cmd)
{
    uint8_t buf[2] = { (cmd >> 8) & 0xFF, cmd & 0xFF };
    return HAL_I2C_Master_Transmit(hi2c, SGP41_I2C_ADDR, buf, 2, HAL_MAX_DELAY);
}

/* write a 2-byte command followed by compensation parameters */
static HAL_StatusTypeDef sgp41_send_cmd_with_params(I2C_HandleTypeDef *hi2c,
                                                      uint16_t cmd,
                                                      uint16_t rh_ticks,
                                                      uint16_t t_ticks)
{
    uint8_t buf[8];
    buf[0] = (cmd >> 8) & 0xFF;
    buf[1] =  cmd & 0xFF;
    buf[2] = (rh_ticks >> 8) & 0xFF;
    buf[3] =  rh_ticks & 0xFF;
    buf[4] = sensirion_crc8(&buf[2], 2);
    buf[5] = (t_ticks >> 8) & 0xFF;
    buf[6] =  t_ticks & 0xFF;
    buf[7] = sensirion_crc8(&buf[5], 2);
    return HAL_I2C_Master_Transmit(hi2c, SGP41_I2C_ADDR, buf, 8, HAL_MAX_DELAY);
}

/* validate a 2-byte word + CRC triplet */
static uint8_t sgp41_check_crc(const uint8_t *buf)
{
    return sensirion_crc8(buf, 2) == buf[2];
}

/* convert RH/T to SGP41 compensation ticks */
static void sgp41_compensation_ticks(float rh_pct, float temp_c,
                                      uint16_t *rh_ticks, uint16_t *t_ticks)
{
    if (rh_pct < 0.0f || temp_c < -45.0f) {
        *rh_ticks = SGP41_DEFAULT_RH_TICKS;
        *t_ticks  = SGP41_DEFAULT_T_TICKS;
    } else {
        *rh_ticks = (uint16_t)(rh_pct  * 65535.0f / 100.0f);
        *t_ticks  = (uint16_t)((temp_c + 45.0f) * 65535.0f / 175.0f);
    }
}

HAL_StatusTypeDef sgp41_conditioning(I2C_HandleTypeDef *hi2c, uint16_t *sraw_voc)
{
    uint8_t buf[3];
    HAL_StatusTypeDef ret;

    ret = sgp41_send_cmd_with_params(hi2c, SGP41_CMD_CONDITIONING,
                                     SGP41_DEFAULT_RH_TICKS,
                                     SGP41_DEFAULT_T_TICKS);
    if (ret != HAL_OK) return ret;

    HAL_Delay(SGP41_DELAY_CONDITIONING_MS);

    ret = HAL_I2C_Master_Receive(hi2c, SGP41_I2C_ADDR, buf, 3, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sgp41_check_crc(buf)) return HAL_ERROR;

    *sraw_voc = ((uint16_t)buf[0] << 8) | buf[1];
    return HAL_OK;
}

HAL_StatusTypeDef sgp41_measure_raw(I2C_HandleTypeDef *hi2c,
                                     float rh_pct, float temp_c,
                                     sgp41_raw_t *out)
{
    uint8_t buf[6];
    uint16_t rh_ticks, t_ticks;
    HAL_StatusTypeDef ret;

    sgp41_compensation_ticks(rh_pct, temp_c, &rh_ticks, &t_ticks);

    ret = sgp41_send_cmd_with_params(hi2c, SGP41_CMD_MEASURE_RAW,
                                     rh_ticks, t_ticks);
    if (ret != HAL_OK) return ret;

    HAL_Delay(SGP41_DELAY_MEASURE_MS);

    ret = HAL_I2C_Master_Receive(hi2c, SGP41_I2C_ADDR, buf, 6, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sgp41_check_crc(&buf[0]) || !sgp41_check_crc(&buf[3])) return HAL_ERROR;

    out->sraw_voc = ((uint16_t)buf[0] << 8) | buf[1];
    out->sraw_nox = ((uint16_t)buf[3] << 8) | buf[4];
    return HAL_OK;
}

HAL_StatusTypeDef sgp41_self_test(I2C_HandleTypeDef *hi2c)
{
    uint8_t buf[3];
    HAL_StatusTypeDef ret;

    ret = sgp41_send_cmd(hi2c, SGP41_CMD_SELF_TEST);
    if (ret != HAL_OK) return ret;

    HAL_Delay(SGP41_DELAY_SELF_TEST_MS);

    ret = HAL_I2C_Master_Receive(hi2c, SGP41_I2C_ADDR, buf, 3, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sgp41_check_crc(buf)) return HAL_ERROR;

    /* bit0 = VOC pixel, bit1 = NOx pixel; 0 = pass */
    return (buf[1] & 0x03) ? HAL_ERROR : HAL_OK;
}

HAL_StatusTypeDef sgp41_get_serial(I2C_HandleTypeDef *hi2c, uint64_t *serial)
{
    uint8_t buf[9];
    HAL_StatusTypeDef ret;

    ret = sgp41_send_cmd(hi2c, SGP41_CMD_SERIAL);
    if (ret != HAL_OK) return ret;

    HAL_Delay(1);

    ret = HAL_I2C_Master_Receive(hi2c, SGP41_I2C_ADDR, buf, 9, HAL_MAX_DELAY);
    if (ret != HAL_OK) return ret;

    if (!sgp41_check_crc(&buf[0]) ||
        !sgp41_check_crc(&buf[3]) ||
        !sgp41_check_crc(&buf[6])) return HAL_ERROR;

    *serial = ((uint64_t)buf[0] << 40) | ((uint64_t)buf[1] << 32) |
              ((uint64_t)buf[3] << 24) | ((uint64_t)buf[4] << 16) |
              ((uint64_t)buf[6] <<  8) |  (uint64_t)buf[7];

    return HAL_OK;
}

HAL_StatusTypeDef sgp41_reset(I2C_HandleTypeDef *hi2c)
{
    /* general call reset — address 0x00, single byte 0x06 */
    uint8_t cmd = 0x06;
    return HAL_I2C_Master_Transmit(hi2c, 0x00, &cmd, 1, HAL_MAX_DELAY);
}
